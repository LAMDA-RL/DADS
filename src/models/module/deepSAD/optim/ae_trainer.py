import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.module.deepSAD.base.base_net import BaseNet
from src.models.module.deepSAD.base.base_trainer import BaseTrainer


# from src.dataset.deep_sad_dataset import SemiSupervisedDataset


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None

    def train(self, train_dataset, ae_net: BaseNet):
        # logger = logging.getLogger()

        # train_dataset = TabularData.load_from_dataframe(train_df,training=True)
        # val_dataset = TabularData.load_from_dataframe(val_df,training=False)

        # Get train data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader,
                                  shuffle=True, drop_last=True)
        # train_loader, _ = train_dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        print('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.float()
                inputs = inputs.to(self.device)

                # import pdb
                # pdb.set_trace()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                  f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        print('Pretraining Time: {:.3f}s'.format(self.train_time))
        print('Finished pretraining.')

        return ae_net

    def test(self, test_dataset, ae_net: BaseNet):
        pass
