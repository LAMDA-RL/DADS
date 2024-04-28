import numpy as np
import pandas as pd
import torch
from pyod.models.iforest import IForest

# from src.models.module.goad.fcnet  import netC1, netC5
from src.models.module.goad.opt_tc_tabular import TransClassifierTabular


class StaticSTOC(object):
    """
    A class for Unsupervised Pyod anomaly detection model for semi-supervised anomaly detection problem


    """

    def __init__(self, config: dict, seed: int, dataset: str):
        """Init unsupervised instance."""

        ## Unpack hyperparameters
        K = config['MODEL']['K']
        # threshold_dict = config['MODEL']['threshold_dict']
        max_samples = config['MODEL']['MAX_SAMPLES']

        # threshold = threshold_dict[dataset]

        model = IForest(max_samples=max_samples, random_state=seed)

        self.K = K
        self.model = model
        # self.threshold = threshold

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, **kwargs):
        """Trains the unsupervised model on the training data."""

        if 'label' in train_df.columns:

            ## Refine data with STOC RefineData block
            train_df = self.data_ensemble_filter(train_df)

            X_train = train_df.drop(['label'], axis=1).values
            y_train = train_df['label'].values



        else:
            train_df = self.data_ensemble_filter(train_df)
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

    def data_ensemble_filter(self, train_df):
        """Filter contaiminated training data with ensemble method"""
        from sklearn.mixture import GaussianMixture

        training_data_list = []
        subset_size = len(train_df) // self.K
        # print("ensemble subset = {}".format(subset_size))
        result_cols = []

        input_cols = train_df.columns.tolist()
        input_cols2 = train_df.columns.tolist()
        if 'class' in input_cols:
            input_cols.remove('class')

        train_X = train_df[input_cols].values

        black_samples_size = len(train_df.loc[train_df['label'] == -1])
        black_threshold = black_samples_size / len(train_df) * 2

        print(train_df['label'].value_counts())

        for idx in range(self.K):

            _train_df = train_df.sample(subset_size)
            _train_X = _train_df[input_cols].values

            # model = HBOS(contamination=contamination)
            # model = IForest(contamination=contamination)
            model = GaussianMixture(n_components=5, covariance_type='full')
            model.fit(_train_X)

            ## 用subset fitted的model 来投票
            # predicted_prob = model.predict_proba(train_X)[:,1]
            predicted_prob = model.score_samples(train_X)
            predicted_prob = predicted_prob * -1  ## GDE的score是多大概率在当前模型， 值越小越是outlier

            if black_threshold == 0:
                black_threshold = 0.0000001

            print("threshold is {}".format(black_threshold))

            threshold = np.percentile(predicted_prob, (1 - black_threshold) * 100)
            filter_result = list(map(lambda x: x > threshold, predicted_prob))
            col = 'ensemble_result_{}'.format(idx)
            result_cols.append(col)
            train_df[col] = filter_result

        ## 取所有weak classifier的结果的并集: 所有classifier都预测是白的才是白的
        train_df['agg_result'] = train_df[result_cols].sum(axis=1)
        result_cols2 = result_cols + ['agg_result']
        # print(train_df['agg_result'].value_counts())
        train_df = train_df.loc[train_df['agg_result'] == 0]
        train_df.reset_index(inplace=True, drop=True)
        train_df.drop(['agg_result'], axis=1, inplace=True)
        train_df = train_df[input_cols2]  ## 只返回原始columns

        # print(train_df['label'].value_counts(normalize=True))

        return train_df

    def save_model(self, export_path):
        """Save Unsupervised model to export_path."""
        pass

    def load_model(self, import_path, device: str = 'cpu'):
        """Load Unsupervised model from import_path."""
        pass


class DynamicSTOC(object):
    """
    A class for Unsupervised Pyod anomaly detection model for semi-supervised anomaly detection problem

    """

    def __init__(self, config: dict, seed: int, dataset: str):

        self.K = 5
        self.n_rots = 256
        self.d_out = 32
        self.epoch = config['TRAIN']['n_epochs']

        self.model = TransClassifierTabular(config, dataset=dataset)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, **kwargs):

        x_train, x_test, y_test_fscore, ratio = self.load_trans_data(train_df, val_df)

        for _epoch in range(self.epoch):

            if _epoch + 1 in [1, 2, 5, 10, 20, 50, 100, 500]:
                print("For epoch {}, train size before refinement: {}".format(_epoch, len(train_df)))
                train_df = self.data_ensemble_filter(train_df)
                print("For epoch {}, train size after refinement: {}".format(_epoch, len(train_df)))

                x_train, x_test, y_test_fscore, ratio = self.load_trans_data(train_df, val_df)
                print("Ratio after data refinement: {}".format(ratio))

            auc_score, auc_pr = self.model.fit_trans_classifier(
                x_train, x_test, y_test_fscore, ratio
            )

        return auc_score, auc_pr

    def load_trans_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, **kwargs):
        """
        GOAD Train
        """

        if 1 in train_df['label'].unique():
            train_df['label'].replace({1: 0}, inplace=True)  ## replace labeled normal class to 0
        if -1 in train_df['label'].unique():
            train_df['label'].replace({-1: 1}, inplace=True)  ## replace labeled anomaly class to 1

        ratio = 100.0 * len(train_df.loc[train_df['label'] == 1]) / len(train_df)

        x_train = train_df.drop(['label'], axis=1).values
        x_test = val_df.drop(['label'], axis=1).values
        y_test = val_df['label'].values

        n_train, n_dims = x_train.shape
        rots = np.random.randn(self.n_rots, n_dims, self.d_out)

        x_train = np.stack([x_train.dot(rot) for rot in rots], 2)
        x_test = np.stack([x_test.dot(rot) for rot in rots], 2)

        return x_train, x_test, y_test, ratio

    def data_ensemble_filter(self, train_df):
        """Filter contaiminated training data with ensemble method"""
        from sklearn.mixture import GaussianMixture

        training_data_list = []
        subset_size = len(train_df) // self.K
        # print("ensemble subset = {}".format(subset_size))
        result_cols = []

        input_cols = train_df.columns.tolist()
        input_cols2 = train_df.columns.tolist()
        if 'class' in input_cols:
            input_cols.remove('class')

        train_X = train_df[input_cols].values

        black_samples_size = len(train_df.loc[train_df['label'] == 1])
        black_threshold = black_samples_size / len(train_df) * 2

        print(train_df['label'].value_counts())

        for idx in range(self.K):

            _train_df = train_df.sample(subset_size)
            _train_X = _train_df[input_cols].values

            # model = HBOS(contamination=contamination)
            # model = IForest(contamination=contamination)
            model = GaussianMixture(n_components=5, covariance_type='full')
            model.fit(_train_X)

            ## 用subset fitted的model 来投票
            # predicted_prob = model.predict_proba(train_X)[:,1]
            predicted_prob = model.score_samples(train_X)
            predicted_prob = predicted_prob * -1  ## GDE的score是多大概率在当前模型， 值越小越是outlier

            if black_threshold == 0:
                black_threshold = 0.0000001

            print("threshold is {}".format(black_threshold))

            threshold = np.percentile(predicted_prob, (1 - black_threshold) * 100)
            filter_result = list(map(lambda x: x > threshold, predicted_prob))
            col = 'ensemble_result_{}'.format(idx)
            result_cols.append(col)
            train_df[col] = filter_result

        ## 取所有weak classifier的结果的并集: 所有classifier都预测是白的才是白的
        train_df['agg_result'] = train_df[result_cols].sum(axis=1)
        result_cols2 = result_cols + ['agg_result']
        # print(train_df['agg_result'].value_counts())
        train_df = train_df.loc[train_df['agg_result'] == 0]
        train_df.reset_index(inplace=True, drop=True)
        train_df.drop(['agg_result'], axis=1, inplace=True)
        train_df = train_df[input_cols2]  ## 只返回原始columns

        # print(train_df['label'].value_counts(normalize=True))

        return train_df

    @staticmethod
    def tc_loss(zs, m):
        means = zs.mean(0).unsqueeze(0)
        res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
        pos = torch.diagonal(res, dim1=1, dim2=2)
        offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
        neg = (res + offset).min(-1)[0]
        loss = torch.clamp(pos + m - neg, min=0).mean()
        return loss


def norm_data(train_real, val_real, val_fake):
    mus = train_real.mean(0)
    sds = train_real.std(0)
    sds[sds == 0] = 1

    def get_norm(xs, mu, sd):
        return np.array([(x - mu) / sd for x in xs])

    train_real = get_norm(train_real, mus, sds)
    val_real = get_norm(val_real, mus, sds)
    val_fake = get_norm(val_fake, mus, sds)
    return train_real, val_real, val_fake


def benchmark():
    """
    Get benchmark dataset from original GOAD paper

    """

    import scipy
    data = scipy.io.loadmat("./data/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    norm_samples = samples[labels == 0]  # 3679 norm
    anom_samples = samples[labels == 1]  # 93 anom

    n_train = len(norm_samples) // 2
    x_train = norm_samples[:n_train]  # 1839 train

    val_real = norm_samples[n_train:]
    val_fake = anom_samples

    train_real, val_real, val_fake = norm_data(x_train, val_real, val_fake)
    y_test_fscore = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])

    ratio = 100.0 * len(val_real) / (len(val_real) + len(val_fake))

    n_train, n_dims = train_real.shape
    rots = np.random.randn(256, n_dims, 32)

    print('Calculating transforms')
    x_train = np.stack([train_real.dot(rot) for rot in rots], 2)
    val_real_xs = np.stack([val_real.dot(rot) for rot in rots], 2)
    val_fake_xs = np.stack([val_fake.dot(rot) for rot in rots], 2)
    x_test = np.concatenate([val_real_xs, val_fake_xs])
    return x_train, x_test, y_test_fscore, ratio
