"""
Top level API for models used in experiments.
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>
from ._models import XgboostAlgo, CatBoostAlgo

###### exposed functionality ##############################################
__all__ = ["XgboostAlgo", "CatBoostAlgo"]

# from .models import (RandomForestAlgo,
#                      AdaBoostAlgo,
#                      LogitBoostAlgo,
#                      GradientBoostAlgo,                           
#                      XgboostAlgo, 
#                      LightGbmAlgo, 
#                      CatBoostAlgo, 
#                      get_models)
###########################################################################

