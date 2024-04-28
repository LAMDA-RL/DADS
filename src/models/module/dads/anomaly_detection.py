import copy
import random

import gym
import numpy as np
import torch
from gym import spaces
from pyod.models.iforest import IForest
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class ad(gym.Env):
    def __init__(self, train_df, valid_df, black_len, white_len, contamination, dataset_name, ground_truth, parameter):
        self.device = parameter["device"]
        self.dataset_name = dataset_name

        self.dataset_anomaly = torch.tensor(train_df.iloc[:black_len, :-1].values.astype(float)).float().to(self.device)
        self.dataset_unlabeled = torch.tensor(
            train_df.iloc[black_len + white_len:, :-1].values.astype(float)).float().to(
            self.device)
        self.dataset = self.dataset_unlabeled
        self.confidence = [0] * len(self.dataset_unlabeled)
        self.anomaly_repeat_times = max(1, round(len(train_df) * parameter["anomaly_ratio"] / black_len))
        for _ in range(self.anomaly_repeat_times):
            self.dataset = torch.cat([self.dataset, self.dataset_anomaly])
            self.confidence = self.confidence + [parameter["check_num"]] * len(self.dataset_anomaly)
        self.valid_df = valid_df
        self.ground_truth = ground_truth
        print("unlabeled: ", len(self.dataset_unlabeled), "abnormal: ", len(self.dataset_anomaly), "*",
              self.anomaly_repeat_times)

        self.current_index = random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[self.current_index]

        self.observation_space = spaces.Discrete(self.current_data.size()[0])
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.tot_steps = 0

        self.sample_num = parameter["sample_num"]
        self.max_trajectory = parameter["max_trajectory"]
        self.check_num = parameter["check_num"]
        self.initial_check_num = parameter["check_num"]
        self.reward_list = parameter["reward_list"]
        self.sampling_method_distribution = parameter["sampling_method_distribution"]
        self.score_threshold = parameter["score_threshold"]
        self.eval_interval = parameter["eval_interval"]
        self.min_steps_before_searching = parameter["min_steps_before_searching"]

        self.up_search_num = min(contamination, 0.04) * self.max_trajectory * parameter["search_percentage"]
        self.low_search_num = self.up_search_num / 2
        print("Search range:", (self.low_search_num, self.up_search_num))

        self.searched_anomalies = 0
        self.correct_search_num = 0
        self.upper_search_num = 0

        self.net = None
        self.best_net = None
        self.critic = None
        self.best_critic = None
        self.critic2 = None
        self.best_critic2 = None
        self.best_roc = 0
        self.best_epoch = -1
        self.tot_reward = 0

        # self.logger = Logger()
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.logger.set_log(SummaryWriter(log_dir="./log/" + current_time))

        self.clf = IForest()
        self.anomaly_score_list = None
        self.pre_sample_method = "random"
        # print("Unsupervised method: ", self.clf.__class__)

        self.touch_num = [0] * (len(self.dataset_unlabeled) + len(self.dataset_anomaly))

    def reset(self):
        if self.tot_steps < self.min_steps_before_searching:
            self.check_num = 100
        else:
            if self.check_num == 100:
                self.check_num = self.initial_check_num
            else:
                if self.searched_anomalies > self.up_search_num:
                    self.check_num = self.check_num + 1
                elif self.searched_anomalies < self.low_search_num:
                    self.check_num = self.check_num - 1
        # print("check num: ", self.check_num)

        self.confidence = [0] * len(self.dataset_unlabeled)
        for _ in range(self.anomaly_repeat_times):
            self.confidence = self.confidence + [self.check_num] * len(self.dataset_anomaly)

        # self.logger.base_idx = self.logger.base_idx + self.max_trajectory

        self.current_index = random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[self.current_index]
        self.pre_sample_method = "random"

        mapped_dataset = self.net.process_hidden_layers(self.dataset)
        self.clf.fit(mapped_dataset.cpu().detach().numpy())
        self.anomaly_score_list = np.array(self.clf.decision_scores_.tolist())
        min_val, max_val = np.min(self.anomaly_score_list), np.max(self.anomaly_score_list)
        self.anomaly_score_list = [(x - min_val) / (max_val - min_val) for x in self.anomaly_score_list]

        # plt.hist(self.anomaly_score_list, bins=100, edgecolor='k')
        # plt.savefig("./figures/score_distribution.jpg")
        # assert 0

        self.touch_num = [0] * (len(self.dataset_unlabeled) + len(self.dataset_anomaly))

        return self.current_data

    def calculate_reward(self, action):
        """ calculate reward based on the class of current data and the action"""
        if self.confidence[self.current_index] >= self.check_num:
            if action == 1:
                score = self.reward_list[0]
            else:
                score = self.reward_list[1]
        elif self.confidence[self.current_index] == self.check_num - 1 and action == 1:
            score = self.reward_list[2] * self.anomaly_score_list[self.current_index]
        elif self.confidence[self.current_index] < self.check_num - 1:
            if self.pre_sample_method == "random" and action == 0:
                score = self.reward_list[3]
            else:
                score = 0
        else:
            score = 0

        return score

    def sampling_function(self, action):
        choice = np.random.choice([i for i in range(2)], size=1,
                                  p=self.sampling_method_distribution)[0]
        if choice == 0:
            self.pre_sample_method = "random"
            self.current_index = random.randint(0, len(self.dataset) - 1)
            self.current_data = self.dataset[self.current_index]
        else:
            self.pre_sample_method = "distance"
            true_sample_num = min(self.sample_num, len(self.dataset))
            candidate_index = np.random.choice([i for i in range(len(self.dataset))], size=true_sample_num,
                                               replace=False)

            with torch.no_grad():
                mapped_current = self.net.process_hidden_layers(self.current_data).cpu()
            if action == 0:
                max_dist = -float('inf')
                for ind in candidate_index:
                    dist = np.linalg.norm(
                        mapped_current - self.net.process_hidden_layers(self.dataset[ind]).detach().cpu())
                    if dist > max_dist:
                        max_dist = dist
                        self.current_data = self.dataset[ind]
                        self.current_index = ind
            else:
                min_dist = float('inf')
                for ind in candidate_index:
                    dist = np.linalg.norm(
                        mapped_current - self.net.process_hidden_layers(self.dataset[ind]).detach().cpu())
                    if dist < min_dist and dist != 0:
                        min_dist = dist
                        self.current_data = self.dataset[ind]
                        self.current_index = ind

        if self.current_index > len(self.dataset_unlabeled):
            touch_index = (self.current_index - len(self.dataset_unlabeled)) % len(self.dataset_anomaly)
        else:
            touch_index = self.current_index
        self.touch_num[touch_index] = self.touch_num[touch_index] + 1

    def refresh(self, score):
        if self.confidence[self.current_index] < self.check_num:
            if score >= self.score_threshold:
                self.confidence[self.current_index] += 1
            elif score < self.score_threshold:
                self.confidence[self.current_index] = 0

    def step(self, s):
        """ Environment takes an action, then returns the current data(regarded as state), reward and done flag"""
        if s > self.score_threshold:
            action = 1
        else:
            action = 0

        reward = self.calculate_reward(action)

        self.tot_steps = self.tot_steps + 1
        self.refresh(s)

        done = False
        if self.tot_steps % self.max_trajectory == 0:
            done = True
            self.searched_anomalies = sum(
                i >= self.check_num for i in self.confidence) - self.anomaly_repeat_times * len(self.dataset_anomaly)

            unlabeled_confidence = self.confidence[:len(self.dataset_unlabeled)]
            self.correct_search_num = sum(
                unlabeled_confidence[i] >= self.check_num and self.ground_truth[i] != 0 for i in
                range(len(unlabeled_confidence)))

            unlabeled_touch = self.touch_num[:len(self.dataset_unlabeled)]
            self.upper_search_num = sum(
                unlabeled_touch[i] >= self.check_num and self.ground_truth[i] != 0 for i in
                range(len(unlabeled_touch)))

        self.sampling_function(action)

        if self.tot_steps % self.eval_interval == 0:
            auc_roc, auc_pr, p95, _ = self.evaluate(self.valid_df, False)
            if auc_roc > self.best_roc:
                self.best_net = copy.deepcopy(self.net)
                self.best_roc = auc_roc
                self.best_critic = copy.deepcopy(self.critic)
                self.best_critic2 = copy.deepcopy(self.critic2)
                self.best_epoch = self.tot_steps / self.max_trajectory

            # self.logger.log("result/auc_roc", auc_roc, self.tot_steps)
            # self.logger.log("result/auc_pr", auc_pr, self.tot_steps)
            # self.logger.log("result/p95", p95, self.tot_steps)

        return self.current_data, reward, done, " "

    def evaluate(self, df, flag):
        """ Evaluate the agent, return AUC_ROC and AUC_PR"""
        x = torch.tensor(df.iloc[:, :-1].values.astype(float)).float().to(self.device)
        y = list(df.iloc[:, -1].values.astype(float))

        if flag:
            q_values = self.best_net(x)
        else:
            q_values = self.net(x)
        anomaly_score = q_values[:, 0]
        # plt.hist(anomaly_score.cpu().detach(), bins=10, color='blue')
        # plt.show()
        # plt.savefig("./figures/distribution"+str(int(self.tot_steps/self.max_trajectory))+".jpg")

        # plt.clf()
        # x_label = [0.1*i for i in range(10)]
        # for temp in x_label:
        #     y_hat = [0 if i < temp * 0.01 else 1 for i in anomaly_score.cpu().detach()]
        #     precision = np.mean([1 if y_hat[i] == y[i] else 0 for i in range(len(y))])
        #     plt.scatter(temp, precision, color='blue')
        # plt.show()
        # plt.savefig("./figures/acc_threshold"+str(int(self.tot_steps/self.max_trajectory))+".jpg")

        auc_roc = roc_auc_score(y, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(y, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)
        fpr, tpr, thresholds = metrics.roc_curve(y, anomaly_score.cpu().detach(), pos_label=1)
        for idx, _tpr in enumerate(tpr):
            if _tpr > 0.95:
                break

        return auc_roc, auc_pr, fpr[idx], self.best_epoch

    def evaluate_unsup(self, clf):
        x = torch.tensor(self.valid_df.iloc[:, :-1].values.astype(float)).float().cpu()
        y = list(self.valid_df.iloc[:, -1].values.astype(float))

        y_hat = [i[1] for i in clf.predict_proba(x)]
        auc_roc = roc_auc_score(y, y_hat)

        return auc_roc
