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
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        data = {"sample": sample, "label": label}
        return data


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')

    return f1


class EarlyStopper:
    def stop(self, epoch, train_loss, valin_loss, testin_loss, test_auc, test_pr, test_p95):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return self.test_loss, self.test_auc, self.test_pr, self.test_p95, self.best_epoch


class Patience(EarlyStopper):
    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=10, use_train_loss=True):
        self.local_val_optimum = float("inf")
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss, self.val_loss, self.test_loss = None, None, None
        self.test_auc, self.test_pr, self.test_p95 = None, None, None

    def stop(self, epoch, train_loss, val_loss, test_loss=None, test_auc=None, test_pr=None, test_p95=None):
        if self.use_train_loss:
            if train_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = train_loss
                self.best_epoch = epoch
                self.train_loss, self.val_loss, self.test_loss = train_loss, val_loss, test_loss
                self.test_auc, self.test_pr, self.test_p95 = test_auc, test_pr, test_p95
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss = train_loss
                self.train_loss, self.val_loss, self.test_loss = train_loss, val_loss, test_loss
                self.test_auc, self.test_pr, self.test_p95 = test_auc, test_pr, test_p95
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
