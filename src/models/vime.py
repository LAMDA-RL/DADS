import json

import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from src.models.module.vime.vime_self import vime_self
from src.models.module.vime.vime_semi import vime_semi


class VIME(object):
    """A class for the Vime semi-supervised training method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, config):
        """Inits DeepSAD with hyperparameter eta."""

        ## Unpack hyperparameters
        hidden_dim = config['MODEL']['hidden_dim']
        self.hidden_dim = hidden_dim
        self.sess = None
        self.model = None

    def set_network(self, ):
        """Builds the neural network """

        pass

    def train(self, train_df, val_df, config):
        """Trains the Deep SAD model on the training data."""

        ## Unpack hyperparameters
        p_m = config['TRAIN']['p_m']
        # alpha = config['PRETRAIN']['alpha']
        K = config['TRAIN']['K']
        beta = config['TRAIN']['beta']
        batch_size = config['TRAIN']['batch_size']
        iterations = config['TRAIN']['iterations']
        pretrain = config['PRETRAIN']['pretrain']

        vime_semi_parameters = dict()
        vime_semi_parameters['hidden_dim'] = self.hidden_dim
        vime_semi_parameters['batch_size'] = batch_size
        vime_semi_parameters['iterations'] = iterations

        ## Replace semi-supervised labeled anomaies -1 to 1
        # train_df = train_df.loc[train_df['label']!=0] ## 0 represents unlabeld data
        # val_df = val_df.loc[val_df['label']!=0] ## 0 represents unlabeld data
        if 1 in train_df['label'].unique():
            train_df['label'].replace({1: 0}, inplace=True)  ## replace labeled normal class to 0
        if -1 in train_df['label'].unique():
            train_df['label'].replace({-1: 1}, inplace=True)  ## replace labeled anomaly class to 1

        if pretrain:
            vime_self_encoder = self.pretrain(train_df, config)

        unlabeled_train_df = train_df.loc[train_df['label'] == 0]
        X_train_unlabeled = unlabeled_train_df.drop(['label'], axis=1).values

        labled_train_df = train_df.loc[train_df['label'] != 0]
        X_train_labeled = labled_train_df.drop(['label'], axis=1).values
        y_train_labeled = labled_train_df['label']

        ## convert to keras one-hot y type
        # n_class = y_train_labeled.nunique()
        n_class = 2
        y_train_labeled = tf.keras.utils.to_categorical(y_train_labeled, num_classes=n_class)

        X_val = val_df.drop(['label'], axis=1).values
        y_val = val_df['label']
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=n_class)  ## by default n_class is 2

        # print(X_train_labeled.shape)
        # print(y_train_labeled.shape)

        y_test_hat = vime_semi(X_train_labeled, y_train_labeled, X_train_unlabeled, X_val,
                               vime_semi_parameters, p_m, K, beta, vime_self_encoder)

        # self.sess = sess
        # self.model = model

        auc_score = roc_auc_score(y_val[:, 1], y_test_hat[:, 1])
        # Compute PR
        precision, recall, _thresholds = precision_recall_curve(y_val[:, 1], y_test_hat[:, 1])
        auc_pr = auc(recall, precision)

        print("validation AUC: {}, PR: {}".format(auc_score, auc_pr))

        # tf.reset_default_graph()
        # tf.keras.backend.clear_session()
        # tf.get_variable_scope().reuse_variables()

        return auc_score, auc_pr

    def evaluate(self, test_df):
        """Tests the VIME model on the test data."""

        # X_test = test_df.drop(['label'],axis=1).values
        # y_test = test_df['label']
        # n_class = y_test.nunique()
        # y_test = tf.keras.utils.to_categorical(y_test, num_classes = n_class) ## by default n_class is 2

        # x_input = tf.placeholder(tf.float32, [None, data_dim])

        #   # Predict on x_test
        # y_test_hat = sess.run(y_hat, feed_dict={x_input: X_test})

        # return auc_score, auc_pr

        pass

    def pretrain(self, train_df, config):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        ## Unpack hyperparameters
        p_m = config['PRETRAIN']['p_m']
        alpha = config['PRETRAIN']['alpha']
        # K = config['PRETRAIN']['K']
        # beta = config['PRETRAIN']['beta']
        device = config['PRETRAIN']['device']
        batch_size = config['PRETRAIN']['batch_size']
        epochs = config['PRETRAIN']['epochs']
        save_dir = config['PRETRAIN']['save_dir']

        unlabeled_train_df = train_df.loc[train_df['label'] == 0]
        X_unlabeled = unlabeled_train_df.drop(['label'], axis=1).values

        vime_self_parameters = dict()
        vime_self_parameters['batch_size'] = batch_size
        vime_self_parameters['epochs'] = epochs

        self.save_dir = save_dir

        vime_self_encoder = vime_self(X_unlabeled, p_m, alpha, vime_self_parameters)
        # vime_self_encoder.save(save_dir)  
        return vime_self_encoder

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
