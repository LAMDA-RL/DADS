import random
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .common import second_to_time_str


class TDReplayBuffer:
    def __init__(self, obs_dim, parameter):
        self.n = parameter["n"]
        self.max_buffer_size = parameter["max_buffer_size"]
        self.gamma = parameter["gamma"]
        action_dim = 1
        self.discrete_action = True
        self.obs_buffer = np.zeros((self.max_buffer_size, obs_dim))
        self.action_buffer = np.zeros((self.max_buffer_size, action_dim))
        self.next_obs_buffer = np.zeros((self.max_buffer_size, obs_dim))
        self.reward_buffer = np.zeros((self.max_buffer_size,))
        self.done_buffer = np.ones((self.max_buffer_size,))
        self.n_step_obs_buffer = np.zeros((self.max_buffer_size, obs_dim))
        self.discounted_reward_buffer = np.zeros((self.max_buffer_size,))
        self.n_step_done_buffer = np.zeros(self.max_buffer_size, )
        self.n_count_buffer = np.ones((self.max_buffer_size,)).astype(np.int) * self.n
        # insert a random state at initialization to avoid bugs when inserting the first state
        self.max_sample_size = 1
        self.curr = 1
        self.device = parameter["device"]

    def add_tuple(self, obs, action, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done

        self.n_step_obs_buffer[self.curr] = next_obs
        self.discounted_reward_buffer[self.curr] = reward
        self.n_step_done_buffer[self.curr] = 0.
        breaked = False
        for i in range(self.n - 1):
            idx = (self.curr - i - 1) % self.max_sample_size
            if self.done_buffer[idx]:
                breaked = True
                break
            self.discounted_reward_buffer[idx] += (self.gamma ** (i + 1)) * reward
        if not breaked and not self.done_buffer[(self.curr - self.n) % self.max_sample_size]:
            self.n_step_obs_buffer[(self.curr - self.n) % self.max_sample_size] = obs
        if done:
            self.n_step_done_buffer[self.curr] = 1.0
            self.n_count_buffer[self.curr] = 1
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 1.0
                self.n_count_buffer[idx] = i + 2
        else:
            prev_idx = (self.curr - 1) % self.max_sample_size
            if not self.done_buffer[prev_idx]:
                self.n_step_done_buffer[prev_idx] = 0.
            for i in range(self.n - 1):
                idx = (self.curr - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = next_obs
                self.n_step_done_buffer[idx] = 0.0

        self.curr = (self.curr + 1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True, n=None):
        if n is not None and n != self.n:
            self.update_td(n)
        if self.done_buffer[self.curr - 1]:
            valid_indices = range(self.max_sample_size)
        elif self.curr >= self.n:
            valid_indices = list(range(self.curr - self.n)) + list(range(self.curr + 1, self.max_sample_size))
        else:
            valid_indices = range(self.curr + 1, self.max_sample_size - (self.n - self.curr))
        batch_size = min(len(valid_indices), batch_size)
        index = random.sample(valid_indices, batch_size)
        obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch = \
            self.obs_buffer[index], \
            self.action_buffer[index], \
            self.n_step_obs_buffer[index], \
            self.discounted_reward_buffer[index], \
            self.n_step_done_buffer[index]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
            if self.discrete_action:
                action_batch = torch.LongTensor(action_batch).to(self.device)
            else:
                action_batch = torch.FloatTensor(action_batch).to(self.device)
            n_step_obs_batch = torch.FloatTensor(n_step_obs_batch).to(self.device)
            discounted_reward_batch = torch.FloatTensor(discounted_reward_batch).to(self.device).unsqueeze(1)
            n_step_done_batch = torch.FloatTensor(n_step_done_batch).to(self.device).unsqueeze(1)

        return obs_batch, action_batch, n_step_obs_batch, discounted_reward_batch, n_step_done_batch

    def update_td(self, n):
        print("Updating the current buffer from td \033[32m{} to {}\033[0m".format(self.n, n))
        self.n_step_obs_buffer = np.zeros_like(self.n_step_obs_buffer)
        self.discounted_reward_buffer = np.zeros_like(self.discounted_reward_buffer)
        self.n_step_done_buffer = np.zeros_like(self.n_step_done_buffer)
        self.mask_buffer = np.zeros_like(self.n_step_done_buffer)
        curr = (self.curr - 1) % self.max_sample_size
        curr_traj_end_idx = curr
        num_trajs = int(np.sum(self.done_buffer))
        if not self.done_buffer[curr]:
            num_trajs += 1
        while num_trajs > 0:
            self.n_step_done_buffer[curr_traj_end_idx] = self.done_buffer[curr_traj_end_idx]
            self.n_step_obs_buffer[curr_traj_end_idx] = self.next_obs_buffer[curr_traj_end_idx]
            curr_traj_len = 1
            idx = (curr_traj_end_idx - 1) % self.max_sample_size
            while not self.done_buffer[idx] and idx != curr:
                idx = (idx - 1) % self.max_sample_size
                curr_traj_len += 1

            for i in range(min(n - 1, curr_traj_len)):
                idx = (curr_traj_end_idx - i - 1) % self.max_sample_size
                if self.done_buffer[idx]:
                    break
                self.n_step_obs_buffer[idx] = self.next_obs_buffer[curr_traj_end_idx]
                self.n_step_done_buffer[idx] = self.done_buffer[curr_traj_end_idx]

            for i in range(curr_traj_len):
                curr_return = self.reward_buffer[(curr_traj_end_idx - i) % self.max_sample_size]
                for j in range(min(n, curr_traj_len - i)):
                    target_idx = curr_traj_end_idx - i - j
                    self.discounted_reward_buffer[target_idx] += (curr_return * (self.gamma ** j))

            if curr_traj_len >= n:
                for i in range(curr_traj_len - n):
                    curr_idx = (curr_traj_end_idx - n - i) % self.max_sample_size
                    if self.done_buffer[curr_idx]:
                        break
                    next_obs_idx = (curr_idx + n) % self.max_sample_size
                    self.n_step_obs_buffer[curr_idx] = self.obs_buffer[next_obs_idx]
            curr_traj_end_idx = (curr_traj_end_idx - curr_traj_len) % self.max_sample_size
            num_trajs -= 1

        self.n = n


class DQNTrainer:
    def __init__(self, agent, env, eval_env, buffer, valid_df, parameter):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.buffer = buffer

        self.batch_size = parameter["batch_size"]
        self.num_steps_per_iteration = parameter["num_steps_per_iteration"]
        self.num_test_trajectories = parameter["num_test_trajectories"]
        self.max_iteration = parameter["max_iteration"]
        self.epsilon = parameter["init_epsilon"]
        self.save_model_interval = parameter["save_model_interval"]
        self.log_interval = parameter["log_interval"]
        self.start_timestep = parameter["start_timestep"]
        self.anneal_rate = (parameter["init_epsilon"] - parameter["final_epsilon"]) / self.num_steps_per_iteration

        self.valid_df = valid_df

        self.parameter = parameter

    def train(self):
        train_traj_rewards = []
        iteration_durations = []
        tot_env_steps = 0

        state = self.env.reset()
        for ite in range(1, self.max_iteration + 1):
            iteration_start_time = time()
            self.epsilon = self.parameter["init_epsilon"]
            traj_reward = 0

            for step in range(self.num_steps_per_iteration):
                if random.random() < self.epsilon:
                    action = random.randint(0, self.env.action_dim - 1)
                else:
                    action = self.agent.select_action(state)
                self.epsilon = self.epsilon - self.anneal_rate

                next_state, reward, done, info = self.env.step(action)
                traj_reward += reward

                self.buffer.add_tuple(state.cpu(), action, next_state.cpu(), reward, float(done))
                state = next_state

                tot_env_steps += 1
                if tot_env_steps < self.start_timestep:
                    continue

                data_batch = self.buffer.sample_batch(self.batch_size)
                self.agent.update(data_batch)

                self.env.refresh_net(self.agent.q_network)

            self.env.refresh_iforest(self.agent.q_network)

            state = self.env.reset()
            train_traj_rewards.append(traj_reward)

            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)

            if ite % self.log_interval == 0:
                avg_test_reward = self.test()
                auc_roc, auc_pr = self.evaluate(self.valid_df)
                remaining_seconds = int((self.max_iteration - ite) * np.mean(iteration_durations[-3:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\tvalid return {:02f}\tauc_roc {:02f}\tauc_pr {:02f}\teta: {}".format(
                    ite, self.max_iteration, train_traj_rewards[-1], avg_test_reward, auc_roc, auc_pr,
                    time_remaining_str)
                print(summary_str)

    def test(self):
        rewards = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            state = self.eval_env.reset()
            for step in range(self.num_steps_per_iteration):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.eval_env.step(action)
                traj_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(traj_reward)
        return np.mean(rewards)

    def evaluate(self, df):
        x = torch.tensor(df.iloc[:, :-1].values.astype(float)).float()
        y = list(df.iloc[:, -1].values.astype(float))

        q_values = self.agent.q_network(x)
        anomaly_score = q_values[:, 1]
        auc_roc = roc_auc_score(y, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(y, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)

        return auc_roc, auc_pr
