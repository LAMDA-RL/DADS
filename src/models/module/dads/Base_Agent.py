import gym
import torch

from .Utility_Functions import NN


class Base_Agent(object):
    """ Base class of agent
    Inherited by SAC_Discrete
    """

    def __init__(self, config, environment):
        self.hyperparameters = config
        self.environment = environment
        # self.action_types = "DISCRETE"
        # self.action_size = int(self.environment.action_space.n)
        self.action_types = "CONTINUOUS"
        self.action_size = 1

        self.state_size = self.environment.current_data.size()[0]
        self.total_episode_score_so_far = 0
        self.episode_number = 0
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warningself.environment.re

    def step(self):
        """ Take a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def eval(self):
        """ Evaluate the agent itself in the game. This method must be overriden by any agent"""
        raise ValueError("Eval needs to be implemented by the agent")

    def pretrain(self):
        """ Evaluate the agent itself in the game. This method must be overriden by any agent"""
        raise ValueError("Pretrain needs to be implemented by the agent")

    def reset_game(self):
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False

    def run_n_episodes(self):
        """ Run game to completion train_round times and then summarises results"""
        while self.episode_number < self.hyperparameters["num_episodes_to_run"]:
            self.reset_game()
            self.step()  # here step functon will complete a single training episode
            val_auc_roc, val_auc_pr, val_p95, _ = self.eval()
            res = "Episode {}: val_auc_roc {:.03f} val_auc_pr {:.03f} val_p95 {:.03f}".format(self.episode_number,
                                                                                              val_auc_roc,
                                                                                              val_auc_pr, val_p95)
            print(res)
            # print("correct/upper searched anomalies/total searched anomalies: {}/{}/{}".format(
            #     self.environment.correct_search_num, self.environment.upper_search_num,
            #     self.environment.searched_anomalies))

        return self.environment.correct_search_num, self.environment.upper_search_num, self.environment.searched_anomalies

    def conduct_action(self, action):
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, hyperparameters=None):
        """ Create a neural network for the agents to use"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"]).to(
            self.environment.device)

    @staticmethod
    def copy_model_over(from_model, to_model):
        """ Copy model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
