"""
Top level API for hyperparameter tuning methods.
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from ._tuning import get_tuner
from ._tuning import tune_model
from ._tuning import OptunaTuner
from ._tuning import HyperOptTuner
from ._tuning import EvolutionarySearchTuner

###### exposed functionality ##############################################
__all__ = ["get_tuner",
           "tune_model",  
           "OptunaTuner",
           "HyperOptTuner", 
           "EvolutionarySearchTuner"]
###########################################################################
