import argparse
import glob
import logging
import logging.config
import os
from datetime import datetime
from os import path as os_path

import pandas as pd
import toml
import yaml
from tqdm import tqdm

from src.datasets.semi_supervised_ad_loader import TabularData
from src.models.builder import BenchmarkBuilder


def setup_log_config(log_path, log_file_name, log_config_file):
    """Set up logger configuration"""

    os.makedirs(log_path, exist_ok=True)
    with open(log_config_file, 'r') as f:
        log_cfg = yaml.safe_load(f.read())
    log_cfg['handlers']['file_handler']['filename'] = os_path.join(
        log_path, log_file_name)
    logging.config.dictConfig(log_cfg)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
# ------------------------------------------------------------------------------#

parser = argparse.ArgumentParser(description='dads')
parser.add_argument('--sample_num', type=int)
parser.add_argument('--max_trajectory', type=int)
parser.add_argument('--check_num', type=int)
parser.add_argument('--search_percentage', type=float)
parser.add_argument('--reward_list', type=float, nargs="+")
parser.add_argument('--sampling_method_distribution', type=float, nargs="+")
parser.add_argument('--anomaly_ratio', type=float)
parser.add_argument('--score_threshold', type=float)
parser.add_argument('--min_steps_before_searching', type=int)
parser.add_argument('--num_episodes_to_run', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--model', type=str, default="dads")
parser.add_argument('--setting', type=str, default="1-1")
parser.add_argument('--test_label', type=str)

args = parser.parse_args()
MODEL_NAME = args.model
setting = args.setting

CONFIG_LIST = glob.glob("./config/{}/*.toml".format(MODEL_NAME))
CONFIG = toml.load(CONFIG_LIST)

change_str = MODEL_NAME + "_" + setting
for item in vars(args):
    if getattr(args, item) is not None and item == "num_episodes_to_run":
        CONFIG["Agent"]["num_episodes_to_run"] = getattr(args, item)
        change_str = change_str + "_" + str(item) + "_" + str(getattr(args, item))
    elif getattr(args, item) is not None and item == "learning_rate":
        CONFIG["Agent"]["Actor"]["learning_rate"] = getattr(args, item)
        CONFIG["Agent"]["Critic"]["learning_rate"] = getattr(args, item)
        change_str = change_str + "_" + str(item) + "_" + str(getattr(args, item))
    elif getattr(args, item) is not None and item != "model" and item != "setting":
        CONFIG["Environment"][str(item)] = getattr(args, item)
        change_str = change_str + "_" + str(item) + "_" + str(getattr(args, item))

print(change_str)

if not os.path.exists("./results"):
    os.makedirs("./results")
## Hyperparameter

## 10% of TOTAL black train data are labeled
## If number of black samples in training dataset is 50, 0.1 means 5 labeled black dataset is available in training set
## the rest of the black dataset(45 out of 50) will be discarded（unless COMTAINATION_RATIO>0）.
ANOMALIES_FRACTION = CONFIG[setting]['SEMI_SUPERVISED_SETTING']['ANOMALIES_FRACTION']

## Labeled normalies to labeled anomalies ratio: 
## Say 5 out of 50 labeled black samples in the training set, NORMALIES_RATIO=5 means 25 labeled normal samples will be in the dataset
## NORMALIES_RATIO=0 means no labeled normal samples in the training dataset.
NORMALIES_RATIO = CONFIG[setting]['SEMI_SUPERVISED_SETTING']['NORMALIES_RATIO']

## Proportion of unlabeled black sampleds in the unlabeled training dataset
## If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.
COMTAINATION_RATIO = CONFIG[setting]['SEMI_SUPERVISED_SETTING']['COMTAINATION_RATIO']


# ------------------------------------------------------------------------------#
#                                 MAIN                                   #
# ------------------------------------------------------------------------------#


def baseline():
    """
    Evaluate semi-supervised model on each dataset
    Iterate over anomalies fraction, normalies_ratio, and comtaination_ratio

    """

    import itertools

    ## Create experiments setting
    benchmark_datasets = CONFIG[setting]['DATA']['DATASETS']
    seeds = CONFIG[setting]['DATA']['SEEDS']

    configs = list(itertools.product(
        benchmark_datasets, seeds, ANOMALIES_FRACTION,
        NORMALIES_RATIO, COMTAINATION_RATIO))

    results = []
    column_name = ['dataset_name', 'seed', 'anomalies_fraction',
                   'normalies_ratio', 'comtaination_ratio', 'roc_auc', 'roc_pr']

    for config in tqdm(configs):

        ## Unpack hyperparameters
        dataset_name, seed, anomalies_fraction, normalies_ratio, comtaination_ratio = config

        logger.info("Current dataset is {}".format(dataset_name))
        logger.info("current setting is {}".format(setting))

        ## Load data
        ad_ds = TabularData.load(dataset_name)
        df = ad_ds._dataset

        ## Semi-supervised setting
        if 'multi' in dataset_name:
            all_normal_classes = CONFIG[setting]['MULTI_CLASS_AD_SETTING']['NORMAL_CLASSES'][dataset_name]
            known_anomaly_class = CONFIG[setting]['MULTI_CLASS_AD_SETTING']['KNOWN_ANOMALY_CLASS'][dataset_name]
            train_df, val_df, test_df, known_val_df, unknown_val_df, known_test_df, unknown_test_df, labeled_length, black_len, white_len, ground_truth, ori_df = TabularData.semi_supervised_multi_class_ad_sampling2(
                df, seed=seed, anomalies_fraction=anomalies_fraction, normalies_ratio=normalies_ratio,
                comtaination_ratio=comtaination_ratio, all_normal_classes=all_normal_classes,
                known_anomaly_class=known_anomaly_class)
        else:
            train_df, val_df, test_df, black_len, white_len, ground_truth, ori_df = TabularData.semi_supervised_ad_sampling(
                df, seed=seed, anomalies_fraction=anomalies_fraction, normalies_ratio=normalies_ratio,
                comtaination_ratio=comtaination_ratio)

        ## Build model
        model = BenchmarkBuilder.build(MODEL_NAME, CONFIG, seed=seed, dataset_name=dataset_name)

        ## Model training
        if MODEL_NAME == 'deepSAD':
            train_dataset = TabularData.load_from_dataframe(train_df, training=True)
            test_dataset = TabularData.load_from_dataframe(test_df, training=False)
            model.train(train_dataset=train_dataset, config=CONFIG)
            roc_auc, roc_pr = model.evaluate(test_dataset)
            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

        elif MODEL_NAME == "vime":
            roc_auc, roc_pr = model.train(train_df=train_df, val_df=test_df, config=CONFIG)
            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])
            del model

        elif MODEL_NAME == "devnet":
            roc_auc, roc_pr = model.train(train_df=train_df, val_df=test_df, config=CONFIG)
            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])
            del model

        elif MODEL_NAME == 'dads':
            model.train(train_df, val_df, black_len, white_len, comtaination_ratio,
                        dataset_name, ground_truth)

            if 'multi' in dataset_name:
                roc_auc, roc_pr, p95, roc_decay, pr_decay, p95_decay, best_episode = \
                    model.evaluate(test_df, known_test_df, unknown_test_df)
                print("Best validation episode: ", best_episode)
                print("test roc: {}, test pr: {}, test p95: {}\n"
                      "roc_decay: {}, pr_decay: {}, p95_decay: {}\n".format(
                        roc_auc, roc_pr, p95, roc_decay, pr_decay, p95_decay))
                results.append([dataset_name, seed, anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc,
                                roc_pr, p95, roc_decay, pr_decay, p95_decay])
                column_name = ['dataset_name', 'seed', 'anomalies_fraction', 'normalies_ratio', 'comtaination_ratio',
                               'roc_auc', 'roc_pr', 'p95', 'roc_decay', 'pr_decay', 'p95_decay']
            else:
                roc_auc, roc_pr, p95, best_episode = model.evaluate(test_df)
                print("Best validation episode: ", best_episode)
                print("test roc: {}, test pr: {}, test p95: {}\n".format(roc_auc, roc_pr, p95))
                results.append([dataset_name, seed,
                                anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr, p95])
                column_name = ['dataset_name', 'seed', 'anomalies_fraction',
                               'normalies_ratio', 'comtaination_ratio', 'roc_auc', 'roc_pr', 'p95']

        elif MODEL_NAME == 'dplan':
            model.train(train_df, val_df, black_len, white_len)
            roc_auc, roc_pr = model.evaluate(test_df)
            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

        elif MODEL_NAME == 'dynamic_stoc':
            roc_auc, roc_pr = model.train(train_df=train_df, val_df=test_df, config=CONFIG)
            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

        elif MODEL_NAME == 'latentoe':
            roc_auc, roc_pr, _ = model.train(train_df, val_df, test_df, black_len, comtaination_ratio)
            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])
        else:
            model.train(train_df, val_df)
            ## Model Evaluation
            roc_auc, roc_pr = model.evaluate(test_df)

            results.append([dataset_name, seed,
                            anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

        ## Save results
        results_df = pd.DataFrame(results)
        results_df.columns = column_name
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        results_df.to_csv("./results/{}.csv".format(change_str), index=False)

    print(change_str)


if __name__ == '__main__':
    baseline()
