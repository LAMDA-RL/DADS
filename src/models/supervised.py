import numpy as np
import pandas as pd


class XGB(object):
    """
    A class for Supervised XGB classification model for semi-supervised anomaly detection problem
    Keep DeepSAD benchmark supervised model setting: assign ALL unlabeled samples ARE normal


    """

    def __init__(self, config: dict, seed: int):
        """Init SSAD instance."""

        ## Unpack hyperparameters
        learning_rate = config['MODEL']['LEARNING_RATE']
        n_estimators = config['MODEL']['N_ESTIMATORS']
        max_depth = config['MODEL']['MAX_DEPTH']

        ## xgb model
        import xgboost as xgb
        model = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, silent=True,
                                  max_depth=max_depth, min_child_weight=1, gamma=0, subsample=0.8,
                                  colsample_bytree=0.8, num_classes=2,
                                  objective='binary:logistic', random_state=seed, missing=np.nan)

        model.n_classes_ = 2

        self.model = model

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, **kwargs):
        """Trains the SSAD model on the training data."""

        ## Replace semi-supervised labeled anomaies -1 to 1
        # train_df = train_df.loc[train_df['label']!=0] ## 0 represents unlabeld data
        # val_df = val_df.loc[val_df['label']!=0] ## 0 represents unlabeld data

        if 1 in train_df['label'].unique():
            train_df['label'].replace({1: 0}, inplace=True)  ## replace labeled normal class to 0
        if -1 in train_df['label'].unique():
            train_df['label'].replace({-1: 1}, inplace=True)  ## replace labeled anomaly class to 1

        X_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values

        X_val = val_df.drop(['label'], axis=1).values
        y_val = val_df['label'].values

        ## Add normal label into training data
        # if train_df['label'].nunique() == 1 and 0 not in train_df['label'].unique(): 
        #     n_feat = X_train.shape[1]
        #     X_train = np.vstack((X_train,np.random.rand(1,n_feat))) 
        #     y_train = np.append(y_train,np.array(0)) 

        # print("unique y+train")
        # print(np.unique(y_train))

        # Use validation set for early stop strategies
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            # n_classes_ = 2
        )

    def evaluate(self, test_df: pd.DataFrame):
        """Tests the SSAD model on the test data."""

        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        X_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values

        # Test on test set
        scores = self.model.predict_proba(X_test)[:, 1]  # outlier labels (0 or 1)

        # Compute AUC
        auc_score = roc_auc_score(y_test, scores)

        # Compute PR
        precision, recall, _thresholds = precision_recall_curve(y_test, scores)
        auc_pr = auc(recall, precision)

        return auc_score, auc_pr

    def save_model(self, export_path):
        """Save XGB model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load XGB model from import_path."""
        pass
