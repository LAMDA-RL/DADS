import torch.nn.functional as F
from torch import nn

from .common import *


class VNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims, use_batch_norm=False):
        super(VNetwork, self).__init__()
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_cls = nn.ReLU
        out_act_cls = nn.Identity
        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = nn.Linear(curr_shape, next_shape)
            if use_batch_norm:
                bn_layer = torch.nn.BatchNorm1d(hidden_dims[i + 1])
                self.networks.extend([curr_network, act_cls(), bn_layer])
            else:
                self.networks.extend([curr_network, act_cls()])
        final_network = nn.Linear(hidden_dims[-1], out_dim)
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.ModuleList(self.networks)

    def forward(self, state):
        out = state
        for i, layer in enumerate(self.networks):
            out = layer(out)
        return out

    def map(self, state):
        out = state
        for i, layer in enumerate(self.networks):
            if i > 1:
                break
            out = layer(out)
        return out


class DQNAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, parameter):
        super(DQNAgent, self).__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.q_target_network = VNetwork(self.obs_dim, self.action_dim, parameter["hidden_dims"])
        self.q_network = VNetwork(self.obs_dim, self.action_dim, parameter["hidden_dims"])

        self.q_optimizer = torch.optim.SGD(self.q_network.parameters(), lr=parameter["learning_rate"],
                                           momentum=parameter["momentum"])

        hard_update_network(self.q_network, self.q_target_network)

        self.q_target_network = self.q_target_network.to(parameter["device"])
        self.q_network = self.q_network.to(parameter["device"])

        self.gamma = parameter["gamma"]
        self.tau = parameter["tau"]
        self.update_target_network_interval = parameter["update_target_network_interval"]
        self.tot_num_updates = 0
        self.n = parameter["n"]

        self.parameter = parameter

    def update(self, data_batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = data_batch

        with torch.no_grad():
            q_target_values = self.q_target_network(next_state_batch)
            q_target_values, q_target_actions = torch.max(q_target_values, dim=1)
            q_target_values = q_target_values.unsqueeze(1)
            q_target = reward_batch + (1. - done_batch) * (self.gamma ** self.n) * q_target_values

        q_current_values = self.q_network(state_batch)
        q_current = torch.gather(q_current_values, 1, action_batch)
        loss = F.mse_loss(q_target, q_current)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.tot_num_updates += 1
        self.try_update_target_network()

    def try_update_target_network(self):
        if self.tot_num_updates % self.update_target_network_interval == 0:
            hard_update_network(self.q_network, self.q_target_network)

    def select_action(self, obs):
        ob = obs.clone().detach().to(self.parameter["device"]).unsqueeze(0).float()
        q_values = self.q_network(ob)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()[0]
        return action

    def save_model(self, target_dir, ite):
        target_dir = os.path.join(target_dir, "ite_{}".format(ite))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        save_path = os.path.join(target_dir, "q_network.pt")
        torch.save(self.q_network, save_path)

    def load_model(self, model_dir):
        q_network_path = os.path.join(model_dir, "q_network.pt")
        self.q_network = torch.load(q_network_path)
