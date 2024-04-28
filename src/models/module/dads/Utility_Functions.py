import copy
import os
import random
from abc import ABC, abstractmethod
from abc import ABCMeta
from collections import namedtuple, deque

from torch.distributions import Categorical, normal


def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


def create_actor_distribution(action_types, actor_output, action_size):
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # This creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:, action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution


class Replay_Buffer(object):
    """
    Replay buffer to store past experiences that the agent can then use for training data
    """

    def __init__(self, buffer_size, batch_size, device=None):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        states, next_states = states.cpu(), next_states.cpu()
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array(
            [np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


def set_random_seeds(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


class RSAMPLE:
    def __init__(self):
        self.decision_scores_ = []

    def fit(self, data):
        self.decision_scores_ = np.array([random.random() for _ in range(len(data))])

    def predict_proba(self, data):
        if len(data.shape) == 1:
            return [[0, random.random()]]
        else:
            return [[0, random.random()] for i in range(data.shape[0])]


class OU_Noise_Exploration():
    """Ornstein-Uhlenbeck noise process exploration strategy"""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.noise = OU_Noise(size, mu, theta, sigma)

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()


import torch
import numpy as np
import torch.nn as nn


#
# rewrite  README using same method as https://github.com/IDSIA/sacred
#
# Example
# -------
# +------------------------------------------------+--------------------------------------------+
# | **Script to train an SVM on the iris dataset** | **The same script as a Sacred experiment** |
# +------------------------------------------------+--------------------------------------------+
# | .. code:: python                               | .. code:: python                           |
# |                                                |                                            |
# |  from numpy.random import permutation          |   from numpy.random import permutation     |
# |  from sklearn import svm, datasets             |   from sklearn import svm, datasets        |
# |                                                |   from sacred import Experiment            |
# |                                                |   ex = Experiment('iris_rbf_svm')          |
# |                                                |                                            |
# |                                                |   @ex.config                               |
# |                                                |   def cfg():                               |
# |  C = 1.0                                       |     C = 1.0                                |
# |  gamma = 0.7                                   |     gamma = 0.7                            |
# |                                                |                                            |
# |                                                |   @ex.automain                             |
# |                                                |   def run(C, gamma):                       |
# |  iris = datasets.load_iris()                   |     iris = datasets.load_iris()            |
# |  perm = permutation(iris.target.size)          |     per = permutation(iris.target.size)    |
# |  iris.data = iris.data[perm]                   |     iris.data = iris.data[per]             |
# |  iris.target = iris.target[perm]               |     iris.target = iris.target[per]         |
# |  clf = svm.SVC(C, 'rbf', gamma=gamma)          |     clf = svm.SVC(C, 'rbf', gamma=gamma)   |
# |  clf.fit(iris.data[:90],                       |     clf.fit(iris.data[:90],                |
# |          iris.target[:90])                     |             iris.target[:90])              |
# |  print(clf.score(iris.data[90:],               |     return clf.score(iris.data[90:],       |
# |                  iris.target[90:]))            |                      iris.target[90:])     |
# +------------------------------------------------+--------------------------------------------+


class Overall_Base_Network(ABC):

    def __init__(self, input_dim, layers_info, output_activation, hidden_activations, dropout, initialiser, batch_norm,
                 y_range, random_seed):

        self.set_all_random_seeds(random_seed)
        self.input_dim = input_dim
        self.layers_info = layers_info

        self.hidden_activations = hidden_activations
        self.output_activation = output_activation
        self.dropout = dropout
        self.initialiser = initialiser
        self.batch_norm = batch_norm
        self.y_range = y_range

        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()

        self.check_all_user_inputs_valid()

        self.initialiser_function = self.str_to_initialiser_converter[initialiser.lower()]

        self.hidden_layers = self.create_hidden_layers()
        self.output_layers = self.create_output_layers()
        self.dropout_layer = self.create_dropout_layer()
        if self.batch_norm: self.batch_norm_layers = self.create_batch_norm_layers()

    @abstractmethod
    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        raise NotImplementedError

    @abstractmethod
    def create_hidden_layers(self):
        """Creates the hidden layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_output_layers(self):
        """Creates the output layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def create_dropout_layer(self):
        """Creates the dropout layers in the network"""
        raise NotImplementedError

    @abstractmethod
    def print_model_summary(self):
        """Prints a summary of the model"""
        raise NotImplementedError

    @abstractmethod
    def set_all_random_seeds(self, random_seed):
        """Sets all random seeds"""
        raise NotImplementedError

    def check_NN_layers_valid(self):
        """Checks that user input for hidden_units is valid"""
        assert isinstance(self.layers_info, list), "hidden_units must be a list"
        list_error_msg = "neurons must be a list of integers"
        integer_error_msg = "Every element of hidden_units must be 1 or higher"
        activation_error_msg = "The number of output activations provided should match the number of output layers"
        for neurons in self.layers_info[:-1]:
            assert isinstance(neurons, int), list_error_msg
            assert neurons > 0, integer_error_msg
        output_layer = self.layers_info[-1]
        if isinstance(output_layer, list):
            assert len(output_layer) == len(self.output_activation), activation_error_msg
            for output_dim in output_layer:
                assert isinstance(output_dim, int), list_error_msg
                assert output_dim > 0, integer_error_msg
        else:
            assert isinstance(self.output_activation, str) or self.output_activation is None, activation_error_msg
            assert isinstance(output_layer, int), list_error_msg
            assert output_layer > 0, integer_error_msg

    def check_NN_input_dim_valid(self):
        """Checks that user input for input_dim is valid"""
        assert isinstance(self.input_dim, int), "input_dim must be an integer"
        assert self.input_dim > 0, "input_dim must be 1 or higher"

    def check_activations_valid(self):
        """Checks that user input for hidden_activations and output_activation is valid"""
        valid_activations_strings = self.str_to_activations_converter.keys()
        if self.output_activation is None: self.output_activation = "None"
        if isinstance(self.output_activation, list):
            for activation in self.output_activation:
                if activation is not None:
                    assert activation.lower() in set(
                        valid_activations_strings), "Output activations must be string from list {}".format(
                        valid_activations_strings)
        else:
            assert self.output_activation.lower() in set(
                valid_activations_strings), "Output activation must be string from list {}".format(
                valid_activations_strings)
        assert isinstance(self.hidden_activations, str) or isinstance(self.hidden_activations,
                                                                      list), "hidden_activations must be a string or a list of strings"
        if isinstance(self.hidden_activations, str):
            assert self.hidden_activations.lower() in set(
                valid_activations_strings), "hidden_activations must be from list {}".format(valid_activations_strings)
        elif isinstance(self.hidden_activations, list):
            assert len(self.hidden_activations) == len(
                self.layers_info), "if hidden_activations is a list then you must provide 1 activation per hidden layer"
            for activation in self.hidden_activations:
                assert isinstance(activation, str), "hidden_activations must be a string or list of strings"
                assert activation.lower() in set(
                    valid_activations_strings), "each element in hidden_activations must be from list {}".format(
                    valid_activations_strings)

    def check_embedding_dimensions_valid(self):
        """Checks that user input for embedding_dimensions is valid"""
        assert isinstance(self.embedding_dimensions, list), "embedding_dimensions must be a list"
        for embedding_dim in self.embedding_dimensions:
            assert len(embedding_dim) == 2 and isinstance(embedding_dim, list), \
                "Each element of embedding_dimensions must be of form (input_dim, output_dim)"

    def check_y_range_values_valid(self):
        """Checks that user input for y_range is valid"""
        if self.y_range:
            assert isinstance(self.y_range, tuple) and len(
                self.y_range) == 2, "y_range must be a tuple of 2 floats or integers"
            for elem in range(2):
                assert isinstance(self.y_range[elem], float) or isinstance(self.y_range[elem],
                                                                           int), "y_range must be a tuple of 2 floats or integers"
            assert self.y_range[0] <= self.y_range[1], "y_range's first element must be smaller than the second element"

    def check_timesteps_to_output_valid(self):
        """Checks that user input for timesteps_to_output is valid"""
        assert self.timesteps_to_output in ["all", "last"]

    def check_initialiser_valid(self):
        """Checks that user input for initialiser is valid"""
        valid_initialisers = set(self.str_to_initialiser_converter.keys())
        assert isinstance(self.initialiser, str), "initialiser must be a string from list {}".format(valid_initialisers)
        assert self.initialiser.lower() in valid_initialisers, "initialiser must be from list {}".format(
            valid_initialisers)

    def check_return_final_seq_only_valid(self):
        """Checks whether user input for return_final_seq_only is a boolean and therefore valid. Only relevant for RNNs"""
        assert isinstance(self.return_final_seq_only, bool)

    def get_activation(self, activations, ix=None):
        """Gets the activation function"""
        if isinstance(activations, list):
            return self.str_to_activations_converter[str(activations[ix]).lower()]
        return self.str_to_activations_converter[str(activations).lower()]


class Base_Network(Overall_Base_Network, ABC):
    """Base class for PyTorch neural network classes"""

    def __init__(self, input_dim, layers_info, output_activation,
                 hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed):
        self.str_to_activations_converter = self.create_str_to_activations_converter()
        self.str_to_initialiser_converter = self.create_str_to_initialiser_converter()
        super().__init__(input_dim, layers_info, output_activation,
                         hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed)
        self.initialise_all_parameters()
        # Flag we use to run checks on the input data into forward the first time it is entered
        self.checked_forward_input_data_once = False

    @abstractmethod
    def initialise_all_parameters(self):
        """Initialises all the parameters of the network"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_data):
        """Runs a forward pass of the network"""
        raise NotImplementedError

    @abstractmethod
    def check_input_data_into_forward_once(self, input_data):
        """Checks the input data into the network is of the right form. Only runs the first time data is provided
        otherwise would slow down training too much"""
        raise NotImplementedError

    def set_all_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

    def create_str_to_activations_converter(self):
        """Creates a dictionary which converts strings to activations"""
        str_to_activations_converter = {"elu": nn.ELU(), "hardshrink": nn.Hardshrink(), "hardtanh": nn.Hardtanh(),
                                        "leakyrelu": nn.LeakyReLU(), "logsigmoid": nn.LogSigmoid(), "prelu": nn.PReLU(),
                                        "relu": nn.ReLU(), "relu6": nn.ReLU6(), "rrelu": nn.RReLU(), "selu": nn.SELU(),
                                        "sigmoid": nn.Sigmoid(), "softplus": nn.Softplus(),
                                        "logsoftmax": nn.LogSoftmax(),
                                        "softshrink": nn.Softshrink(), "softsign": nn.Softsign(), "tanh": nn.Tanh(),
                                        "tanhshrink": nn.Tanhshrink(), "softmin": nn.Softmin(),
                                        "softmax": nn.Softmax(dim=1),
                                        "none": None}
        return str_to_activations_converter

    def create_str_to_initialiser_converter(self):
        """Creates a dictionary which converts strings to initialiser"""
        str_to_initialiser_converter = {"uniform": nn.init.uniform_, "normal": nn.init.normal_,
                                        "eye": nn.init.eye_,
                                        "xavier_uniform": nn.init.xavier_uniform_, "xavier": nn.init.xavier_uniform_,
                                        "xavier_normal": nn.init.xavier_normal_,
                                        "kaiming_uniform": nn.init.kaiming_uniform_,
                                        "kaiming": nn.init.kaiming_uniform_,
                                        "kaiming_normal": nn.init.kaiming_normal_, "he": nn.init.kaiming_normal_,
                                        "orthogonal": nn.init.orthogonal_, "default": "use_default"}
        return str_to_initialiser_converter

    def create_dropout_layer(self):
        """Creates a dropout layer"""
        return nn.Dropout(p=self.dropout)

    def create_embedding_layers(self):
        """Creates the embedding layers in the network"""
        embedding_layers = nn.ModuleList([])
        for embedding_dimension in self.embedding_dimensions:
            input_dim, output_dim = embedding_dimension
            embedding_layers.extend([nn.Embedding(input_dim, output_dim)])
        return embedding_layers

    def initialise_parameters(self, parameters_list):
        """Initialises the list of parameters given"""
        initialiser = self.str_to_initialiser_converter[self.initialiser.lower()]
        if initialiser != "use_default":
            for parameters in parameters_list:
                if type(parameters) == nn.Linear or type(parameters) == nn.Conv2d:
                    initialiser(parameters.weight)
                elif type(parameters) in [nn.LSTM, nn.RNN, nn.GRU]:
                    initialiser(parameters.weight_hh_l0)
                    initialiser(parameters.weight_ih_l0)

    def flatten_tensor(self, tensor):
        """Flattens a tensor of shape (a, b, c, d, ...) into shape (a, b * c * d * .. )"""
        return tensor.reshape(tensor.shape[0], -1)

    def print_model_summary(self):
        print(self)


class NN(nn.Module, Base_Network):
    """Creates a PyTorch neural network
    Args:
        - input_dim: Integer to indicate the dimension of the input into the network
        - layers_info: List of integers to indicate the width and number of linear layers you want in your network,
                      e.g. [5, 8, 1] would produce a network with 3 linear layers of width 5, 8 and then 1
        - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
                              (not including the output layer). Default is ReLU.
        - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                             strings if you want multiple output heads
        - dropout: Float to indicate what dropout probability you want applied after each hidden layer
        - initialiser: String to indicate which initialiser you want used to initialise all the parameters. All PyTorch
                       initialisers are supported. PyTorch's default initialisation is the default.
        - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default is False
        - columns_of_data_to_be_embedded: List to indicate the columns numbers of the data that you want to be put through an embedding layer
                                          before being fed through the other layers of the network. Default option is no embeddings
        - embedding_dimensions: If you have categorical variables you want embedded before flowing through the network then
                                you specify the embedding dimensions here with a list like so: [ [embedding_input_dim_1, embedding_output_dim_1],
                                [embedding_input_dim_2, embedding_output_dim_2] ...]. Default is no embeddings
        - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
                   output values to in regression tasks. Default is no range restriction
        - random_seed: Integer to indicate the random seed you want to use
    """

    def __init__(self, input_dim, layers_info, output_activation=None,
                 hidden_activations="relu", dropout=0.0, initialiser="default", batch_norm=False,
                 columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range=(), random_seed=0):
        nn.Module.__init__(self)
        self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        self.embedding_dimensions = embedding_dimensions
        self.embedding_layers = self.create_embedding_layers()
        Base_Network.__init__(self, input_dim, layers_info, output_activation, hidden_activations, dropout, initialiser,
                              batch_norm, y_range, random_seed)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_NN_input_dim_valid()
        self.check_NN_layers_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def create_hidden_layers(self):
        """Creates the linear layers in the network"""
        linear_layers = nn.ModuleList([])
        input_dim = int(self.input_dim - len(self.embedding_dimensions) + np.sum(
            [output_dims[1] for output_dims in self.embedding_dimensions]))
        for hidden_unit in self.layers_info[:-1]:
            linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            input_dim = hidden_unit
        return linear_layers

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = nn.ModuleList([])
        if len(self.layers_info) >= 2:
            input_dim = self.layers_info[-2]
        else:
            input_dim = self.input_dim
        if not isinstance(self.layers_info[-1], list):
            output_layer = [self.layers_info[-1]]
        else:
            output_layer = self.layers_info[-1]
        for output_dim in output_layer:
            output_layers.extend([nn.Linear(input_dim, output_dim)])
        return output_layers

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(num_features=hidden_unit) for hidden_unit in self.layers_info[:-1]])
        return batch_norm_layers

    def initialise_all_parameters(self):
        """Initialises the parameters in the linear and embedding layers"""
        self.initialise_parameters(self.hidden_layers)
        self.initialise_parameters(self.output_layers)
        self.initialise_parameters(self.embedding_layers)

    def forward(self, x):
        """Forward pass for the network"""
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x)
        if self.embedding_to_occur: x = self.incorporate_embeddings(x)
        x = self.process_hidden_layers(x)
        out = self.process_output_layers(x)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * nn.Sigmoid()(out)
        return out

    def check_input_data_into_forward_once(self, x):
        """Checks the input data into forward is of the right format. Then sets a flag indicating that this has happened once
        so that we don't keep checking as this would slow down the model too much"""
        for embedding_dim in self.columns_of_data_to_be_embedded:
            data = x[:, embedding_dim]
            data_long = data.long()
            assert all(data_long >= 0), "All data to be embedded must be integers 0 and above -- {}".format(data_long)
            assert torch.sum(abs(data.float() - data_long.float())) < 0.0001, """Data columns to be embedded should be integer 
                                                                                values 0 and above to represent the different 
                                                                                classes"""
        if self.input_dim > len(self.columns_of_data_to_be_embedded):
            assert isinstance(x, torch.FloatTensor) or isinstance(x,
                                                                  torch.cuda.FloatTensor), "Input data must be a float tensor"
        assert len(x.shape) == 2, "X should be a 2-dimensional tensor: {}".format(x.shape)
        self.checked_forward_input_data_once = True  # So that it doesn't check again

    def incorporate_embeddings(self, x):
        """Puts relevant data through embedding layers and then concatenates the result with the rest of the data ready
        to then be put through the linear layers"""
        all_embedded_data = []
        for embedding_layer_ix, embedding_var in enumerate(self.columns_of_data_to_be_embedded):
            data = x[:, embedding_var].long()
            embedded_data = self.embedding_layers[embedding_layer_ix](data)
            all_embedded_data.append(embedded_data)
        all_embedded_data = torch.cat(tuple(all_embedded_data), dim=1)
        x = torch.cat((x[:,
                       [col for col in range(x.shape[1]) if col not in self.columns_of_data_to_be_embedded]].float(),
                       all_embedded_data), dim=1)
        return x

    def process_hidden_layers(self, x):
        """Puts the data x through all the hidden layers"""
        for layer_ix, linear_layer in enumerate(self.hidden_layers):
            x = self.get_activation(self.hidden_activations, layer_ix)(linear_layer(x))
            if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
            if self.dropout != 0.0: x = self.dropout_layer(x)
        return x

    def process_output_layers(self, x):
        """Puts the data x through all the output layers"""
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            temp_output = output_layer(x)
            if activation is not None: temp_output = activation(temp_output)
            if out is None:
                out = temp_output
            else:
                out = torch.cat((out, temp_output), dim=1)
        return out


class Singleton:
    def __new__(cls, *arg, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class Logger(Singleton):
    logger = None

    def __init__(self):
        self.base_idx = 0

    def set_log(self, logger):
        self.logger = logger

    def log(self, key, value, idx):
        if self.logger is not None:
            self.logger.add_scalar(key, value, idx + self.base_idx)
