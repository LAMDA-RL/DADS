import json

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_kernels

from src.models.module.ssad.ssad_convex import ConvexSSAD


class SSAD(object):
    """
    A class for kernel SSAD models as described in Goernitz et al., Towards Supervised Anomaly Detection, JAIR, 2013.
    """

    def __init__(self, config: dict):
        """Init SSAD instance."""

        ## Unpack hyperparameters
        kernel = config['MODEL']['kernel']
        kappa = config['MODEL']['kappa']
        Cp = config['MODEL']['cp']
        Cu = config['MODEL']['cu']
        Cn = config['MODEL']['cn']
        gammas = config['MODEL']['gammas']

        self.kernel = kernel
        self.kappa = kappa
        self.Cp = Cp
        self.Cu = Cu
        self.Cn = Cn
        self.rho = None
        self.gammas = gammas

        self.best_gamma = None
        self.best_model = None
        self.best_svs = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, **kwargs):
        """Trains the SSAD model on the training data."""

        X_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values

        X_val = val_df.drop(['label'], axis=1).values
        y_val = val_df['label'].values

        # Training
        print('Starting training...')

        best_auc = 0.0

        ## Use validation df to finetune gamma
        if val_df is not None:

            for _idx, gamma in enumerate(self.gammas):

                # Build the training kernel
                kernel = pairwise_kernels(X_train, X_train, metric=self.kernel, gamma=gamma)

                # Model candidate
                model = ConvexSSAD(kernel, y_train, Cp=self.Cp, Cu=self.Cu, Cn=self.Cn)

                # Train
                model.fit()

                # Test on small hold-out set from test set
                kernel_val = pairwise_kernels(X_val, X_train[model.svs, :], metric=self.kernel, gamma=gamma)
                scores = (-1.0) * model.apply(kernel_val)
                scores = scores.flatten()

                # Compute AUC
                auc = roc_auc_score(y_val, scores)

                print("Gamma: {} Val AUC: {}".format(gamma, auc))

                if auc > best_auc:
                    best_auc = auc
                    self.best_model = model
                    self.best_gamma = gamma

            # Get support vectors for testing
            self.best_svs = X_train[self.best_model.svs, :]

        print(f'Best Model: | Gamma: {self.best_gamma:.8f} | AUC: {100. * best_auc:.2f}')
        print('Finished training.')

    def evaluate(self, test_df: pd.DataFrame):
        """Tests the SSAD model on the test data."""

        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        X_test = test_df.drop(['label'], axis=1)
        y_test = test_df['label'].values

        # Test on test set
        kernel_test = pairwise_kernels(X_test, self.best_svs, metric=self.kernel, gamma=self.best_gamma)
        scores = (-1.0) * self.best_model.apply(kernel_test)
        scores = scores.flatten()

        # Compute AUC
        auc_score = roc_auc_score(y_test, scores)

        # Compute PR
        precision, recall, _thresholds = precision_recall_curve(y_test, scores)
        auc_pr = auc(recall, precision)

        return auc_score, auc_pr

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load SSAD model from import_path."""
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
