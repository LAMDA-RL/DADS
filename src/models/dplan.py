import torch

from src.models.module.dplan.dqn_agent import DQNAgent
from src.models.module.dplan.dqn_trainer import DQNTrainer
from src.models.module.dplan.dqn_trainer import TDReplayBuffer
from src.models.module.dplan.environment import Environment


class dplan(object):
    def __init__(self, config):
        """Init DADS instance."""
        self.parameter = config

    def train(self, train_df, valid_df, black_len, white_len):
        device = self.parameter["device"]
        dataset_a = torch.tensor(train_df.iloc[:black_len, :-1].values.astype(float)).float().to(device)
        dataset_u = torch.tensor(train_df.iloc[black_len + white_len:, :-1].values.astype(float)).float().to(
            device)

        env = Environment(dataset_a, dataset_u, self.parameter)
        eval_env = Environment(dataset_a, dataset_u, self.parameter)

        buffer = TDReplayBuffer(env.obs_dim, self.parameter)

        agent = DQNAgent(env.obs_dim, env.action_dim, self.parameter)

        env.refresh_net(agent.q_network)
        eval_env.refresh_net(agent.q_network)
        env.refresh_iforest(agent.q_network)
        eval_env.refresh_iforest(agent.q_network)

        self.trainer = DQNTrainer(agent, env, eval_env, buffer, valid_df, self.parameter)

        self.trainer.train()

    def evaluate(self, test_df):
        auc_roc, auc_pr = self.trainer.evaluate(test_df)

        return auc_roc, auc_pr

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device):
        """Load SSAD model from import_path."""
        pass
