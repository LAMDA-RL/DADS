# -*- coding: utf-8 -*-

"""Example code for the Builder Module. This code is meant
just for illustrating basic anomaly detection models

Update this when you start working on your own Anomaly Detection project.
"""

import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------#
#                                 MODULE                                   #
# ------------------------------------------------------------------------------#

from .ssad import SSAD
from .supervised import XGB
from .DeepSAD import DeepSAD
from .unsupervised import UnsupervisedModel
from .devnet import DevNet
from .vime import VIME
from .dads import DADS
from .stoc import StaticSTOC, DynamicSTOC
from .dplan import dplan
from .latentoe import LatentOE


# ------------------------------------------------------------------------------#
#                                 Config                                    #
# ------------------------------------------------------------------------------#


# ------------------------------------------------------------------------------#
#                                MODEL BUILDER                                  #
# ------------------------------------------------------------------------------#

class BenchmarkBuilder(object):
    """
    Class represents that a builder to build dummy classification model
    """

    @staticmethod
    def build(model_name, config, **kwargs):
        """
        Build a XLMR multi-modal late concat model.

        Arguments:
        ---------
            model_name: model to build

        Returns
        -------
            model: benchmark model
                

        """

        if model_name == 'ssad':
            model = SSAD(config)
        elif model_name == "supervised":
            seed = kwargs.get('seed')
            model = XGB(config, seed=seed)
        elif model_name == 'deepSAD':
            seed = kwargs.get('seed')
            dataset_name = kwargs.get('dataset_name')
            model_name = dataset_name + "_mlp"

            model = DeepSAD(config)
            model.set_network(model_name)

        elif model_name == 'unsupervised':
            seed = kwargs.get('seed')
            model = UnsupervisedModel(config, seed=seed)
        elif model_name == 'devnet':
            seed = kwargs.get('seed')
            model = DevNet(config, seed=seed)
        elif model_name == 'vime':
            model = VIME(config)
        elif model_name == "dads":
            model = DADS(config, kwargs.get('dataset_name'))
        elif model_name == "dplan":
            model = dplan(config)
        elif model_name == "static_stoc":
            seed = kwargs.get('seed')
            dataset_name = kwargs.get('dataset_name')
            model = StaticSTOC(config, seed=seed, dataset=dataset_name)
        elif model_name == "dynamic_stoc":
            seed = kwargs.get('seed')
            dataset_name = kwargs.get('dataset_name')
            model = DynamicSTOC(config, seed=seed, dataset=dataset_name)
        elif model_name == "latentoe":
            model = LatentOE(config)
        else:
            assert 0

        return model
