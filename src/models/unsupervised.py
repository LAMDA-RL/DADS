import pandas as pd
from pyod.models.iforest import IForest


class UnsupervisedModel(object):
    """
    A class for Unsupervised Pyod anomaly detection model for semi-supervised anomaly detection problem


    """

    def __init__(self, config: dict, seed: int):
        """Init unsupervised instance."""

        ## Unpack hyperparameters
        model_type = config['MODEL']['MODEL_TYPE']

        if model_type == 'isolation_forest':
            max_samples = config['MODEL']['MAX_SAMPLES']
            model = IForest(max_samples=max_samples, random_state=seed)

        self.model = model

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, **kwargs):
        """Trains the unsupervised model on the training data."""

        if 'label' in train_df.columns:

            X_train = train_df.drop(['label'], axis=1).values
            y_train = train_df['label'].values

        else:
            X_train = train_df.values

        self.model.fit(X_train)

    def evaluate(self, test_df: pd.DataFrame):
        """Tests the unsupervised model on the test data."""

        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        X_test = test_df.drop(['label'], axis=1)
        y_test = test_df['label']

        # Test on test set
        scores = self.model.predict_proba(
            X_test.values, method='linear')[:, 1]  # outlier labels (0 or 1)

        # Compute AUC
        auc_score = roc_auc_score(y_test, scores)

        # Compute PR
        precision, recall, _thresholds = precision_recall_curve(y_test, scores)
        auc_pr = auc(recall, precision)

        return auc_score, auc_pr

    def save_model(self, export_path):
        """Save Unsupervised model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load Unsupervised model from import_path."""
        pass
