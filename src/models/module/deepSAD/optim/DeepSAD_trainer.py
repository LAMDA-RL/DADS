import time

import numpy as np
import torch
import torch.optim as optim
# # from src.datasets.semi_supervised_ad_loader import TabularData
# # import sys
# # sys.path.append("../../../src/")
# import src
# from src.datasets.semi_supervised_ad_loader import TabularData
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.utils.data.dataloader import DataLoader

from src.models.module.deepSAD.base.base_net import BaseNet
from src.models.module.deepSAD.base.base_trainer import BaseTrainer


class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, use_hsc=False):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.use_hsc = use_hsc

    def train(self, train_dataset, net: BaseNet):

        # train_dataset = TabularData.load_from_dataframe(train_df,training=True)
        # val_dataset = TabularData.load_from_dataframe(val_df,training=False)

        # Get train data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader,
                                  shuffle=True, drop_last=True)
        # train_loader, _ = train_dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            print('Center c initialized.')

        # Training
        print('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, semi_targets = data
                inputs = inputs.float()
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)

                if not self.use_hsc:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    losses = torch.where(semi_targets == 0, dist,
                                         self.eta * ((dist + self.eps) ** semi_targets.float()))
                    loss = torch.mean(losses)
                else:
                    ## May 10: Apply HSC loss from https://github.com/lukasruff/Classification-AD/blob/master/src/optim/classifier_trainer.py
                    ## May 12 DW: 添加labeled 白样本监督信号到loss里
                    dist = torch.sqrt(torch.sum((outputs - self.c) ** 2, dim=1) + 1) - 1
                    scores = 1 - torch.exp(-dist)
                    # losses = torch.where(semi_targets == 0, dist, -torch.log(scores + self.eps))
                    losses = torch.where(semi_targets == 0, dist, semi_targets.float() * torch.log(scores + self.eps))
                    loss = torch.mean(losses)
                    # dists = torch.sqrt(torch.norm((outputs-self.c), p=2, dim=1) ** 2 + 1) - 1

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                  f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        print('Training Time: {:.3f}s'.format(self.train_time))
        print('Finished training.')

        return net

    def test(self, test_dataset, net: BaseNet, evaluation=True):

        # test_dataset = TabularData.load_from_dataframe(test_df,training=False)
        # val_dataset = TabularData.load_from_dataframe(val_df,training=False)

        # Get train data loader
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader,
                                 shuffle=False, drop_last=False)
        # train_loader, _ = train_dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        print('Starting testing...')
        start_time = time.time()
        result = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.float()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = net(inputs)
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                # loss = torch.mean(losses)
                # scores = dist

                if not self.use_hsc:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)

                else:
                    ## May 10: Apply HSC loss from https://github.com/lukasruff/Classification-AD/blob/master/src/optim/classifier_trainer.py
                    dist = torch.sqrt(torch.sum((outputs - self.c) ** 2, dim=1) + 1) - 1

                scores = dist
                # Save triples of (idx, label, score) in a list
                if evaluation:
                    result += list(zip(labels.cpu().data.numpy().tolist(),
                                       scores.cpu().data.numpy().tolist()))
                else:
                    result += list(zip(scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time

        # Compute AUC
        if evaluation:
            labels, scores = zip(*result)
            labels = np.array(labels)
            scores = np.array(scores)

            # Compute AUC
            auc_score = roc_auc_score(labels, scores)

            # Compute PR
            precision, recall, _thresholds = precision_recall_curve(labels, scores)
            auc_pr = auc(recall, precision)

            # Log results
            print('Test AUC: {:.2f}%'.format(100. * auc_score))
            print('Test Time: {:.3f}s'.format(self.test_time))
            print('Finished testing.')

            return auc_score, auc_pr
        else:
            scores = zip(*result)
            scores = np.array(scores)
            return scores

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.float()
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
