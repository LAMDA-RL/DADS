# -*- coding: utf-8 -*-

"""
Apr 30: 对Semi-supervised AD实验重新设计的Loader


"""

# ------------------------------------------------------------------------------#
#                                 MODULE                                   #
# ------------------------------------------------------------------------------#

import numpy as np
import pandas as pd

# from __init__ import logger
from torch.utils.data import Dataset as PytorchDataset

# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
# ------------------------------------------------------------------------------#

## Hyperparameter
seed = 42

## 10% of TOTAL black train data are labeled
## If number of black samples in training dataset is 50, 0.1 means 5 labeled black dataset is available in training set
## the rest of the black dataset(45 out of 50) will be discarded（unless COMTAINATION_RATIO>0）.
ANOMALIES_FRACTION = 0.1

## Labeled normalies to labeled anomalies ratio: 
## Say 5 out of 50 labeled black samples in the training set, NORMALIES_RATIO=5 means 25 labeled normal samples will be in the dataset
## NORMALIES_RATIO=0 means no labeled normal samples in the training dataset.
NORMALIES_RATIO = 0

## Proportion of unlabeled black sampleds in the unlabeled training dataset
## If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.
COMTAINATION_RATIO = 0.01


# ------------------------------------------------------------------------------#
#                                 MAIN                                   #
# ------------------------------------------------------------------------------#


class TabularData(PytorchDataset):
    """

    The class represents Tabular Anomlay Detection Dataset Module.
    The Dataset is inherited from PytorchDataset and CSVDataSets

    Arguments:
    ---------
        dataset_name(str):
            Dataset name
        _dataset(pd.DataFrame):
            pd.Dataframe includes raw image path, processed image path, ETL results.
            For training, ground truth is saved in "MUBADA" column as well.

    Methods:
    --------

        from_path: 
            Load image data from path
        sanity_check: 
            input data sanity check

    Note: Split/Sanity Check/Consistency Check/Feature Check functions can be user-defined

    """

    def __init__(self,
                 dataset=None,
                 dataset_name='',
                 training=False,
                 target_col_list=None,
                 numerical_feat_idx='',
                 categorical_feat_idx=''
                 ):

        self._dataset = dataset
        self.dataset_name = dataset_name
        self.numerical_feat_idx = numerical_feat_idx
        self.categorical_feat_idx = categorical_feat_idx
        self.target_col_list = target_col_list
        self.training = training

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):

        if self.training:  ## Training placeholder
            if 'label' in self._dataset.columns:
                X = self._dataset.iloc[idx].drop(['label']).values
                y = self._dataset.iloc[idx]['label']

                return X, y
            else:
                return self._dataset.iloc[idx], None
        else:
            if 'label' in self._dataset.columns:
                X = self._dataset.iloc[idx].drop(['label']).values
                y = self._dataset.iloc[idx]['label']

                return X, y
            else:
                return self._dataset.iloc[idx], None

    def set_label(self, label=0):
        if 'label' in self._dataset.columns:
            print("Label will be overwritten to {}".format(label))

        self._dataset['label'] = label

    @classmethod
    def load_from_dataframe(cls, df, training=True):
        """
        Load from pd.Dataframe
        """
        return cls(dataset=df, training=training)

    @classmethod
    def load(cls, dataset_name='arrhythmia', training=True):
        """
        Load Anomaly Detection Benchmark datasets

        """

        if dataset_name in ['arrhythmia', 'annthyroid', 'cardio', 'shuttle', 'satimage2', 'satellite', 'thyroid']:
            df = pd.read_csv("./data/{}.csv".format(dataset_name))

            ## unify label class name
            if 'label' not in df.columns and 'Class' in df.columns:
                df.rename({"Class": "label"}, axis=1, inplace=True)

        elif dataset_name in ['multi_annthyroid', 'multi_cardio',
                              'multi_covertype', 'multi_har', 'multi_shuttle']:

            df = pd.read_csv("./data/multi_class_ad/{}.csv".format(dataset_name.split('_')[-1]))

            if dataset_name == 'multi_cardio':
                df.drop(['CLASS'], axis=1, inplace=True)

            ## unify label class name
            if 'label' not in df.columns and 'Class' in df.columns:
                df.rename({"Class": "label"}, axis=1, inplace=True)
            if 'label' not in df.columns and 'NSP' in df.columns:
                df.rename({"NSP": "label"}, axis=1, inplace=True)

        return cls(dataset=df,
                   dataset_name=dataset_name, numerical_feat_idx='',
                   categorical_feat_idx='', training=training)

    def binary_anomaly_detection_label(self):
        """
        Make multi-class anomaly detection dataset as binary label dataset
        0 represents normal class and 1 represents 1 class
        """

        pass

    @staticmethod
    def semi_supervised_ad_sampling(
            df, seed=42, anomalies_fraction=0.1, normalies_ratio=5, comtaination_ratio=0.01):

        """
        PU Learning Setting for semi-supervised anomaly detection experiment
        
        对数据集按80/20 分train/test
        对分好的train dataset里的黑标， 按anomalies_fraction*black_label_size，当作labeled 数据(labeled数据只可能有黑标)
        对分好的train dataset里的白标， 添加一定比例的黑标做为unlabeled dataset
        最后的train_dataset是上两步的集合(labeled black + unlabeled white + unlabeled black(optional))
        test dataset 还是一开始分到的20%原始数据
        
        Arguments:
        ---------
            anomalies_fraction(float): 
                按anomalies_fraction*black_label_size，采样黑标当作labeled 数据
            comtaination_ratio(float):
                对unlabeled train dataset, 采样train dataset里白标数量*comtaination_ratio的黑标添加到unlabeled train dataset
                e.g: train dataset 有100个白标and comtaination_ratio=0.05, 最后的unlabeled train dataset会是100白标+5黑标=105
                Note: comtaination_ratio超过原始数据黑标浓度话，会替换成原始黑标浓度
            normalies_ratio(int):
                Proportion of unlabeled black sampleds in the unlabeled training dataset
                If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.


        returns:
        --------
            train_df: 按semi supervised AD setting采样后的train df
            test_df: 原始数据的20% stratified split, by default training/test split is 20%


        """

        import random
        from sklearn.model_selection import train_test_split

        random.seed(seed)
        np.random.seed(seed)

        ## check comtaination rate
        odds = df['label'].sum() / len(df)  ## 原始数据黑标浓度

        y = df[['label']]
        X = df.drop(['label'], axis=1)

        ## Step 1: Stratified sampling
        ## May 05: Add validation split for hyperparameter tuning
        ## By defaule 0.6,0.2,0.2
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, stratify=y, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, stratify=y_val_test, test_size=0.5, random_state=seed)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler().fit(X_train)
        X_train_stand = pd.DataFrame(scaler.transform(X_train))
        X_val_stand = pd.DataFrame(scaler.transform(X_val))
        X_test_stand = pd.DataFrame(scaler.transform(X_test))

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = pd.DataFrame(minmax_scaler.transform(X_train_stand))
        X_val_scaled = pd.DataFrame(minmax_scaler.transform(X_val_stand))
        X_test_scaled = pd.DataFrame(minmax_scaler.transform(X_test_stand))

        y_train.reset_index(inplace=True, drop=True)
        y_val.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)

        train_df = pd.concat([X_train_scaled, y_train], axis=1)
        val_df = pd.concat([X_val_scaled, y_val], axis=1)
        test_df = pd.concat([X_test_scaled, y_test], axis=1)

        ## Step 2: Set up labeled positive label size with anomalies_fraction hyperparameters
        black_train_df = train_df.loc[train_df['label'] == 1]
        white_train_df = train_df.loc[train_df['label'] == 0]

        black_train_df_shuffled = black_train_df.sample(frac=1, random_state=seed)
        white_train_df_shuffled = white_train_df.sample(frac=1, random_state=seed)

        black_train_size = len(black_train_df_shuffled)
        labeled_black_sample_size = int(anomalies_fraction * black_train_size)
        labeled_black_sample_size = max(1, labeled_black_sample_size)  ## 最少都需要有一个labeled anomalies
        labeled_black_train_df = black_train_df_shuffled.iloc[:labeled_black_sample_size]

        ## 用DeepSAD&SSAD对semi-supervised label的setting
        ## labeled black samples标-1，labeled white samples标+1
        ## unlabeled samples 标0
        labeled_black_train_df['label'] = -1

        if normalies_ratio > 0:
            ## 假如labeled training黑样本=5，ratio=5, 则labeled white size 应该是25
            labeled_white_size = int(labeled_black_sample_size * normalies_ratio)
            labeled_white_train_df = white_train_df_shuffled.iloc[:labeled_white_size]
            labeled_white_train_df['label'] = 1
            unlabeled_white_train_df = white_train_df_shuffled.iloc[labeled_white_size:]
            labeled_train_df = pd.concat([labeled_black_train_df, labeled_white_train_df])
        else:
            unlabeled_white_train_df = white_train_df_shuffled
            labeled_train_df = labeled_black_train_df

        ## Step 3: For rest of the black data in the training set, 
        ## use comtaination_ratio hyperparameter to add into unlabeled training set
        ## Note comtaination_ratio cannot be larger than odds(如果原数据最多就10%黑的，那comtaination_ratio不可能大于10%)

        comtaination_ratio = min(comtaination_ratio, odds)
        unlabeled_black_train_df = black_train_df_shuffled.iloc[labeled_black_sample_size:]
        white_train_size = len(white_train_df)
        unlabeled_black_train_size = int(white_train_size * comtaination_ratio)

        ## Add those into ublabeled data
        unlabeled_black_train_df2 = unlabeled_black_train_df.iloc[:unlabeled_black_train_size]

        ## Add unlabled black adata into unlabeld white train data
        unlabeled_train_df = pd.concat([unlabeled_black_train_df2, unlabeled_white_train_df])
        ground_truth = list(unlabeled_train_df.iloc[:, -1])
        unlabeled_train_df['label'] = 0

        ## Step 4: finally: concat labeled black training data and comtainnated unlabeled data as final train size
        train_df2 = pd.concat([labeled_train_df, unlabeled_train_df])

        train_df2.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        return train_df2, val_df, test_df, labeled_black_sample_size, int(
            labeled_black_sample_size * normalies_ratio), ground_truth, train_df

    @classmethod
    def concat_dataset(cls, dataset1, dataset2):
        """Concat two pd datasets together"""

        result_df = pd.concat([dataset1._dataset, dataset2._dataset])

        return cls(dataset=result_df)

    @staticmethod
    def semi_supervised_multi_class_ad_sampling(
            df, known_anomaly_class, seed=42, anomalies_fraction=0.1, normalies_ratio=5, comtaination_ratio=0.01,
            all_normal_classes=[]):

        """
        PU Learning Setting for semi-supervised multi-class anomaly detection experiment
        
        对数据集按80/20 分train/test
        对分好的train dataset里的黑标， 按anomalies_fraction*black_label_size，当作labeled 数据(labeled数据只可能有黑标)
        black_label_size是基于入参的known anomaly class
        test dataset里的黑标是所有的anomaly class

        对分好的train dataset里的白标， 添加一定比例的黑标(所有anomaly class都有可能)做为unlabeled dataset
        最后的train_dataset是上两步的集合(labeled black + unlabeled white + unlabeled black(optional))
        test dataset 还是一开始分到的20%原始数据
        
        Arguments:
        ---------
            known_anomaly_class(int):
                训练数据集包含的anomaly 的类别
            anomalies_fraction(float): 
                按anomalies_fraction*black_label_size，采样黑标当作labeled 数据
            comtaination_ratio(float):
                对unlabeled train dataset, 采样train dataset里白标数量*comtaination_ratio的黑标添加到unlabeled train dataset
                e.g: train dataset 有100个白标and comtaination_ratio=0.05, 最后的unlabeled train dataset会是100白标+5黑标=105
                Note: comtaination_ratio超过原始数据黑标浓度话，会替换成原始黑标浓度
            normalies_ratio(int):
                Proportion of unlabeled black sampleds in the unlabeled training dataset
                If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.
            all_normal_classes(list):
                normal classes labels in the dataset


        returns:
        --------
            train_df: 按semi supervised AD setting采样后的train df
            test_df: 原始数据的20% stratified split, by default training/test split is 20%


        """

        import random
        from sklearn.model_selection import train_test_split

        random.seed(seed)
        np.random.seed(seed)

        ## rename normal class labels
        ## use label==0 to subset normal class latter
        df['label'] = df['label'].apply(lambda x: 0 if x in all_normal_classes else x)

        ## check comtaination rate
        # odds = df['label'].sum()/len(df)  ## 原始数据黑标浓度
        odds = 1 - len(df.loc[df['label'] == 0]) / len(df)  ## 原始数据黑标浓度, 为 1-白标浓度

        y = df[['label']]
        X = df.drop(['label'], axis=1)

        ## Step 1: Stratified sampling
        ## May 05: Add validation split for hyperparameter tuning
        ## By defaule 0.6,0.2,0.2
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, stratify=y, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, stratify=y_val_test, test_size=0.5, random_state=seed)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler().fit(X_train)
        X_train_stand = pd.DataFrame(scaler.transform(X_train))
        X_val_stand = pd.DataFrame(scaler.transform(X_val))
        X_test_stand = pd.DataFrame(scaler.transform(X_test))

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = pd.DataFrame(minmax_scaler.transform(X_train_stand))
        X_val_scaled = pd.DataFrame(minmax_scaler.transform(X_val_stand))
        X_test_scaled = pd.DataFrame(minmax_scaler.transform(X_test_stand))

        y_train.reset_index(inplace=True, drop=True)
        y_val.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)

        train_df = pd.concat([X_train_scaled, y_train], axis=1)
        val_df = pd.concat([X_val_scaled, y_val], axis=1)
        test_df = pd.concat([X_test_scaled, y_test], axis=1)

        ## Step 2: Set up labeled positive label size with anomalies_fraction hyperparameters
        known_black_train_df = train_df.loc[train_df['label'] == known_anomaly_class]
        white_train_df = train_df.loc[train_df['label'] == 0]

        black_train_df_shuffled = known_black_train_df.sample(frac=1, random_state=seed)
        white_train_df_shuffled = white_train_df.sample(frac=1, random_state=seed)

        black_train_size = len(black_train_df_shuffled)
        labeled_black_sample_size = int(anomalies_fraction * black_train_size)
        labeled_black_sample_size = max(1, labeled_black_sample_size)  ## 最少都需要有一个labeled anomalies
        labeled_black_train_df = black_train_df_shuffled.iloc[:labeled_black_sample_size]

        ## 用DeepSAD&SSAD对semi-supervised label的setting
        ## labeled black samples标-1， labeled white samples标+1
        ## unlabeled samples 标0

        labeled_black_train_df['label'] = -1

        if normalies_ratio > 0:
            ## 假如labeled training黑样本=5，ratio=5, 则labeled white size 应该是25
            labeled_white_size = int(labeled_black_sample_size * normalies_ratio)
            labeled_white_train_df = white_train_df_shuffled.iloc[:labeled_white_size]
            labeled_white_train_df['label'] = 1
            unlabeled_white_train_df = white_train_df_shuffled.iloc[labeled_white_size:]
            labeled_train_df = pd.concat([labeled_black_train_df, labeled_white_train_df])
        else:
            unlabeled_white_train_df = white_train_df_shuffled
            labeled_train_df = labeled_black_train_df

        ## Step 3: For rest of the black data in the training set, 
        ## use comtaination_ratio hyperparameter to add into unlabeled training set
        ## Note comtaination_ratio cannot be larger than odds(如果原数据最多就10%黑的，那comtaination_ratio不可能大于10%)
        ## 污染的黑标可以来自任何anomaly classes

        comtaination_ratio = min(comtaination_ratio, odds)
        all_black_train_df = train_df.loc[train_df['label'] != 0]

        unlabeled_black_train_df = all_black_train_df.drop(labeled_black_train_df.index,
                                                           axis=0)  ## exclude labeled anomalies
        unlabeled_black_train_df_shuffled = unlabeled_black_train_df.sample(frac=1, random_state=seed)

        white_train_size = len(white_train_df)
        unlabeled_black_train_size = int(white_train_size * comtaination_ratio)
        unlabeled_black_train_size = min(unlabeled_black_train_size, len(unlabeled_black_train_df_shuffled))  ## 确保不溢出

        ## Add those into ublabeled data
        unlabeled_black_train_df2 = unlabeled_black_train_df_shuffled.iloc[:unlabeled_black_train_size]

        ## Add unlabled black adata into unlabeld white train data
        unlabeled_train_df = pd.concat([unlabeled_black_train_df2, unlabeled_white_train_df])
        ground_truth = list(unlabeled_train_df.iloc[:, -1])
        unlabeled_train_df['label'] = 0

        ## Step 4: finally: concat labeled black training data and comtainnated unlabeled data as final train size
        train_df2 = pd.concat([labeled_train_df, unlabeled_train_df])

        train_df2.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        ## For multi-class evaluation, make multi-class label binary
        val_df['label'] = val_df['label'].apply(lambda x: 0 if x == 0 else 1)
        test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 0 else 1)

        return train_df2, val_df, test_df, labeled_black_sample_size, int(
            labeled_black_sample_size * normalies_ratio), ground_truth, train_df

    def semi_supervised_multi_class_ad_sampling2(
            df, known_anomaly_class, seed=42, anomalies_fraction=0.1, normalies_ratio=5, comtaination_ratio=0.01,
            all_normal_classes=[], deepSAD_mode=False):

        """
        PU Learning Setting for semi-supervised multi-class anomaly detection experiment

        对数据集按80/20 分train/test
        对分好的train dataset里的黑标， 按anomalies_fraction*black_label_size，当作labeled 数据(labeled数据只可能有黑标)
        black_label_size是基于入参的known anomaly class
        test dataset里的黑标是所有的anomaly class

        对分好的train dataset里的白标， 添加一定比例的黑标(所有anomaly class都有可能)做为unlabeled dataset
        最后的train_dataset是上两步的集合(labeled black + unlabeled white + unlabeled black(optional))
        test dataset 还是一开始分到的20%原始数据

        Arguments:
        ---------
            known_anomaly_class(int):
                训练数据集包含的anomaly 的类别
            anomalies_fraction(float):
                按anomalies_fraction*black_label_size，采样黑标当作labeled 数据
            comtaination_ratio(float):
                对unlabeled train dataset, 采样train dataset里白标数量*comtaination_ratio的黑标添加到unlabeled train dataset
                e.g: train dataset 有100个白标and comtaination_ratio=0.05, 最后的unlabeled train dataset会是100白标+5黑标=105
                Note: comtaination_ratio超过原始数据黑标浓度话，会替换成原始黑标浓度
            normalies_ratio(int):
                Proportion of unlabeled black sampleds in the unlabeled training dataset
                If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.
            all_normal_classes(list):
                normal classes labels in the dataset


        returns:
        --------
            train_df: 按semi supervised AD setting采样后的train df
            test_df: 原始数据的20% stratified split, by default training/test split is 20%


        """

        import random
        from sklearn.model_selection import train_test_split

        random.seed(seed)
        np.random.seed(seed)

        # import pdb
        # pdb.set_trace()

        ## rename normal class labels
        ## use label==0 to subset normal class latter
        df['label'] = df['label'].apply(lambda x: 0 if x in all_normal_classes else x)

        ## check comtaination rate
        # odds = df['label'].sum()/len(df)  ## 原始数据黑标浓度
        odds = 1 - len(df.loc[df['label'] == 0]) / len(df)  ## 原始数据黑标浓度, 为 1-白标浓度

        y = df[['label']]
        X = df.drop(['label'], axis=1)

        ## Step 1: Stratified sampling
        ## May 05: Add validation split for hyperparameter tuning
        ## By defaule 0.6,0.2,0.2
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, stratify=y, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, stratify=y_val_test, test_size=0.5, random_state=seed)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        X_train_stand = pd.DataFrame(scaler.transform(X_train))
        X_val_stand = pd.DataFrame(scaler.transform(X_val))
        X_test_stand = pd.DataFrame(scaler.transform(X_test))

        # Scale to range [0,1]
        # minmax_scaler = MinMaxScaler().fit(X_train_stand)
        # X_train_scaled = pd.DataFrame(minmax_scaler.transform(X_train_stand))
        # X_val_scaled = pd.DataFrame(minmax_scaler.transform(X_val_stand))
        # X_test_scaled = pd.DataFrame(minmax_scaler.transform(X_test_stand))

        y_train.reset_index(inplace=True, drop=True)
        y_val.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)

        train_df = pd.concat([X_train_stand, y_train], axis=1)
        val_df = pd.concat([X_val_stand, y_val], axis=1)
        test_df = pd.concat([X_test_stand, y_test], axis=1)

        ## Step 2: Set up labeled positive label size with anomalies_fraction hyperparameters
        known_black_train_df = train_df.loc[train_df['label'] == known_anomaly_class]
        white_train_df = train_df.loc[train_df['label'] == 0]

        black_train_df_shuffled = known_black_train_df.sample(frac=1, random_state=seed)
        white_train_df_shuffled = white_train_df.sample(frac=1, random_state=seed)

        black_train_size = len(black_train_df_shuffled)
        labeled_black_sample_size = int(anomalies_fraction * black_train_size)
        labeled_black_sample_size = max(1, labeled_black_sample_size)  ## 最少都需要有一个labeled anomalies
        labeled_black_train_df = black_train_df_shuffled.iloc[:labeled_black_sample_size]

        ## 用DeepSAD&SSAD对semi-supervised label的setting
        ## labeled black samples标-1， labeled white samples标+1
        ## unlabeled samples 标0

        labeled_black_train_df['label'] = -1

        if normalies_ratio > 0:
            ## 假如labeled training黑样本=5，ratio=5, 则labeled white size 应该是25
            labeled_white_size = int(labeled_black_sample_size * normalies_ratio)
            labeled_white_train_df = white_train_df_shuffled.iloc[:labeled_white_size]
            labeled_white_train_df['label'] = 1
            unlabeled_white_train_df = white_train_df_shuffled.iloc[labeled_white_size:]
            labeled_train_df = pd.concat([labeled_black_train_df, labeled_white_train_df])
        else:
            unlabeled_white_train_df = white_train_df_shuffled
            labeled_train_df = labeled_black_train_df

        ## Step 3: For rest of the black data in the training set,
        ## use comtaination_ratio hyperparameter to add into unlabeled training set
        ## Note comtaination_ratio cannot be larger than odds(如果原数据最多就10%黑的，那comtaination_ratio不可能大于10%)
        ## 污染的黑标可以来自任何anomaly classes

        comtaination_ratio = min(comtaination_ratio, odds)
        all_black_train_df = train_df.loc[train_df['label'] != 0]

        unlabeled_black_train_df = all_black_train_df.drop(labeled_black_train_df.index,
                                                           axis=0)  ## exclude labeled anomalies
        unlabeled_black_train_df_shuffled = unlabeled_black_train_df.sample(frac=1, random_state=seed)

        white_train_size = len(white_train_df)
        unlabeled_black_train_size = int(white_train_size * comtaination_ratio)
        unlabeled_black_train_size = min(unlabeled_black_train_size, len(unlabeled_black_train_df_shuffled))  ## 确保不溢出

        ## Add those into unlabeled data
        unlabeled_black_train_df2 = unlabeled_black_train_df_shuffled.iloc[:unlabeled_black_train_size]

        ## Add unlabled black adata into unlabeld white train data
        unlabeled_train_df = pd.concat([unlabeled_black_train_df2, unlabeled_white_train_df])
        ground_truth = list(unlabeled_train_df.iloc[:, -1])
        unlabeled_train_df['label'] = 0

        ## Step 4: finally: concat labeled black training data and comtainnated unlabeled data as final train size

        ## fy23 S2: orca需要区分labeled的和unlabeled的
        labeled_length = len(labeled_train_df)
        train_df2 = pd.concat([labeled_train_df, unlabeled_train_df])

        train_df2.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        ## Fy23 Dec 14: 对val, test 分成known unknown 和unknown unknown， 分别计算模型效能
        val_white_df = val_df.loc[val_df['label'] == 0]
        known_black_val_df = val_df.loc[val_df['label'] == known_anomaly_class]
        unknown_black_val_df = val_df.loc[val_df['label'] != known_anomaly_class].loc[val_df['label'] != 0]

        known_val_df = pd.concat([val_white_df, known_black_val_df])
        unknown_val_df = pd.concat([val_white_df, unknown_black_val_df])

        known_val_df.reset_index(inplace=True, drop=True)
        unknown_val_df.reset_index(inplace=True, drop=True)

        test_white_df = test_df.loc[test_df['label'] == 0]
        known_black_test_df = test_df.loc[test_df['label'] == known_anomaly_class]
        unknown_black_test_df = test_df.loc[test_df['label'] != known_anomaly_class].loc[test_df['label'] != 0]

        known_test_df = pd.concat([test_white_df, known_black_test_df])
        unknown_test_df = pd.concat([test_white_df, unknown_black_test_df])

        known_test_df.reset_index(inplace=True, drop=True)
        unknown_test_df.reset_index(inplace=True, drop=True)

        ## For multi-class evaluation, make multi-class label binary
        val_df['label'] = val_df['label'].apply(lambda x: 0 if x == 0 else 1)
        test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 0 else 1)

        known_test_df['label'] = known_test_df['label'].apply(lambda x: 0 if x == 0 else 1)
        unknown_test_df['label'] = unknown_test_df['label'].apply(lambda x: 0 if x == 0 else 1)

        known_val_df['label'] = known_val_df['label'].apply(lambda x: 0 if x == 0 else 1)
        unknown_val_df['label'] = unknown_val_df['label'].apply(lambda x: 0 if x == 0 else 1)

        if not deepSAD_mode:
            train_df2['label'].replace({1: 0}, inplace=True)  ## 如果不是deepSAD能利用labeled白标的话，labeled 白标也是0
            train_df2['label'].replace({-1: 1}, inplace=True)

        return train_df2, val_df, test_df, known_val_df, unknown_val_df, known_test_df, unknown_test_df, labeled_length, labeled_black_sample_size, int(
            labeled_black_sample_size * normalies_ratio), ground_truth, train_df


def test():
    """
    Unit test semi_supervised dataloader

    """

    ## Hyperparameter
    seed = 42
    ANOMALIES_FRACTION = 0.1  ## 10% black train data are labeled
    COMTAINATION_RATIO = 0
    NORMALIES_RATIO = 5

    DATASET = "arrhythmia"

    ad_ds = TabularData.load(DATASET)
    df = ad_ds._dataset

    ## Semi-supervised setting output
    train_df, test_df = TabularData.semi_supervised_ad_sampling(
        df, seed=seed, anomalies_fraction=ANOMALIES_FRACTION
        , normalies_ratio=NORMALIES_RATIO
        , comtaination_ratio=COMTAINATION_RATIO
    )

    print(train_df['label'].value_counts())

# if __name__ == '__main__':
#     test()
