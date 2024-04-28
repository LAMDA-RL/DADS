# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, roc_curve

from src.models.module.latentoe.utils import compute_pre_recall_f1


class NeutralAD_trainer:
    def __init__(self, model, loss_function, config):

        self.loss_fun = loss_function
        self.device = torch.device(config['device'])
        self.model = model.to(self.device)

        self.train_method = config['train_method']
        self.max_epochs = config['training_epochs']
        self.warmup = 2

    def _train(self, epoch, train_loader, optimizer):
        self.model.train()
        loss_all = 0

        for data in train_loader:
            samples = data['sample']
            labels = data['label']
            # samples = samples.to(self.device)

            z = self.model(samples)
            loss_n, loss_a = self.loss_fun(z)

            if epoch <= self.warmup:
                if self.train_method == 'gt':
                    loss = torch.cat([loss_n[labels == 0], loss_a[labels == 1]], 0)
                    loss_mean = loss.mean()
                else:
                    loss = loss_n
                    loss_mean = loss.mean()
            else:
                score = loss_n - loss_a

                if self.train_method == 'blind':
                    loss = loss_n
                    loss_mean = loss.mean()
                elif self.train_method == 'loe_hard':
                    _, idx_n = torch.topk(score, int(score.shape[0] * (1 - self.contamination)), largest=False,
                                          sorted=False)
                    _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True,
                                          sorted=False)
                    loss = torch.cat([loss_n[idx_n], loss_a[idx_a]], 0)
                    loss_mean = loss.mean()
                elif self.train_method == 'loe_soft':
                    _, idx_n = torch.topk(score, int(score.shape[0] * (1 - self.contamination)), largest=False,
                                          sorted=False)
                    _, idx_a = torch.topk(score, int(score.shape[0] * self.contamination), largest=True, sorted=False)
                    loss = torch.cat([loss_n[idx_n], 0.5 * loss_n[idx_a] + 0.5 * loss_a[idx_a]], 0)
                    loss_mean = loss.mean()
                elif self.train_method == 'refine':
                    _, idx_n = torch.topk(loss_n, int(loss_n.shape[0] * (1 - self.contamination)), largest=False,
                                          sorted=False)
                    loss = loss_n[idx_n]
                    loss_mean = loss.mean()
                elif self.train_method == 'gt':
                    loss = torch.cat([loss_n[labels == 0], loss_a[labels == 1]], 0)
                    loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum()

        return loss_all.item() / len(train_loader.dataset)

    def detect_outliers(self, loader):
        model = self.model
        model.eval()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                samples = data['sample']
                labels = data['label']
                # samples = samples.to(self.device)
                z = model(samples)
                loss_n, loss_a = self.loss_fun(z)
                score = loss_n
                loss_in += loss_n[labels == 0].sum()
                loss_out += loss_n[labels == 1].sum()
                target_all.append(labels)
                score_all.append(score)

        score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc_roc = roc_auc_score(target_all, score_all)
        precision, recall, _thresholds = precision_recall_curve(target_all, score_all)
        auc_pr = auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(target_all, score_all, pos_label=1)
        for idx, _tpr in enumerate(tpr):
            if _tpr > 0.95:
                break
        f1 = compute_pre_recall_f1(target_all, score_all)
        ap = average_precision_score(target_all, score_all)
        return auc_roc, auc_pr, fpr[idx], ap, f1, score_all, loss_in.item() / (
                    target_all == 0).sum(), loss_out.item() / (target_all == 1).sum()

    def train(self, train_loader, contamination, query_num=0, optimizer=None, scheduler=None,
              validation_loader=None, test_loader=None, early_stopper=None, logger=None, log_every=100):
        self.contamination = contamination

        test_loss, testin_loss, test_auc, test_pr, test_p95 = None, None, None, None, None

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train(epoch, train_loader, optimizer)

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                test_auc, test_pr, test_p95, test_ap, test_f1, test_score, testin_loss, testout_loss = self.detect_outliers(
                    test_loader)

            if validation_loader is not None:
                val_auc, val_pr, val_p95, val_ap, val_f1, _, valin_loss, valout_loss = self.detect_outliers(
                    validation_loader)
                if epoch > self.warmup:
                    if early_stopper is not None and early_stopper.stop(epoch, train_loss, valin_loss, testin_loss,
                                                                        test_auc, test_pr, test_p95):
                        break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss}, VL auc: {val_auc} VL pr: {val_pr} VL p95: {val_p95} '

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        if early_stopper is not None:
            test_loss, test_auc, test_pr, test_p95, best_epoch = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at epoch {best_epoch}'
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        return test_loss, test_auc, test_pr, test_p95
