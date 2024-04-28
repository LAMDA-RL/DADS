import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.models.module.latentoe.Losses import DCL
from src.models.module.latentoe.NeuralAD_trainer import NeutralAD_trainer
from src.models.module.latentoe.TabNets import TabNets
from src.models.module.latentoe.model import TabNeutralAD
from src.models.module.latentoe.utils import CustomDataset, Logger, Patience


class LatentOE(object):
    def __init__(self, config):
        """Init DADS instance."""
        self.parameter = config["MODEL"]

    def train(self, train_df, valid_df, test_df, black_len, comtaination_ratio):
        train_dataset = np.array(train_df.iloc[:, :-1].values.astype(float))
        train_label = np.zeros([len(train_df), ])
        train_label[:black_len] = 1
        comtaination_ratio = ((len(train_df) - black_len) * comtaination_ratio + black_len) / len(train_df)
        trainset = CustomDataset(train_dataset, train_label)
        valid_dataset = np.array(valid_df.iloc[:, :-1].values.astype(float))
        valid_label = np.array(valid_df.iloc[:, -1].values.astype(float))
        validset = CustomDataset(valid_dataset, valid_label)
        test_dataset = np.array(test_df.iloc[:, :-1].values.astype(float))
        test_label = np.array(test_df.iloc[:, -1].values.astype(float))
        testset = CustomDataset(test_dataset, test_label)

        train_loader = DataLoader(trainset, batch_size=self.parameter['batch_size'], shuffle=self.parameter['shuffle'],
                                  drop_last=False)
        val_loader = DataLoader(validset, batch_size=self.parameter['batch_size'], shuffle=False,
                                drop_last=False)
        test_loader = DataLoader(testset, batch_size=self.parameter['batch_size'], shuffle=False,
                                 drop_last=False)

        model = TabNeutralAD(TabNets(), train_dataset.shape[-1], self.parameter)
        optimizer = Adam(model.parameters(), lr=self.parameter['learning_rate'], weight_decay=self.parameter['l2'])
        scheduler = StepLR(optimizer, step_size=self.parameter['scheduler_step_size'],
                           gamma=self.parameter['scheduler_gamma'])
        trainer = NeutralAD_trainer(model, loss_function=DCL(self.parameter['loss_temp']), config=self.parameter)
        early_stopper = Patience(patience=self.parameter['early_stopper_patience'],
                                 use_train_loss=self.parameter['early_stopper_use_train_loss'])
        logger = Logger('./logs/latentoe_log.txt', mode='a')

        test_loss, test_auc, test_pr, test_p95 = \
            trainer.train(train_loader=train_loader,
                          contamination=comtaination_ratio, query_num=0,
                          optimizer=optimizer, scheduler=scheduler,
                          validation_loader=val_loader, test_loader=test_loader,
                          early_stopper=early_stopper, logger=logger)
        print("auc_roc:", test_auc, "auc_pr:", test_pr, "p95:", test_p95)

        return test_auc, test_pr, test_p95

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device):
        """Load SSAD model from import_path."""
        pass
