"""
Boosting classifiers tested in the experiments.
The following models are included; 
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### Importing dependencies #############################################
import pickle
import numpy as np 
import pandas as pd
from .. import utils as u 
from .. import tune as tu 
from .. import metrics as ev 
from abc import ABC, abstractmethod
from collections import OrderedDict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Boosting models 
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO -> Model fitted check & Validation where neeeded 
# TODO -> Persistence layer 

###### Constants #########################################################
MODEL_BASE  = "model_base"
MODEL_RF    = "random_forest"
MODEL_XGB   = "xg_boost"
MODEL_LIGHT = "light_boost"
MODEL_CAT   = "cat_boost"

###### Base classifier ###################################################
class _BaseAlgo(ABC, BaseEstimator, ClassifierMixin):
    """
    Base class for models.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    # Constructor ---------------------------------------------------------
    @abstractmethod
    def __init__(self, 
                 tune_props=None, 
                 **kwargs):
        self._model = None 
        self._model_name = MODEL_BASE
        self._tuner = None 
        self._tune_props = tune_props

    # Properties ----------------------------------------------------------
    @property
    def model_name_(self):
        return self._model_name
    
    @property
    def feature_importances_(self):
        return pd.DataFrame(self._model.feature_importances_,
                            index = self._features,
                            columns=["importance"]).sort_values(
                                "importance", 
                                ascending=False)

    @property
    def tune_results_(self):
        if self._tune_props is None:
            raise TypeError("'tune_props not passed'")
        check_is_fitted(self._model, ["feature_importances_"])
        return (self._tuner.meta_results_, self._tuner.results_)

    # Parameters functions ------------------------------------------------
    def set_params(self, **params):
        return self._model.set_params(**params)
    
    def get_params(self, deep=True):
        return self._model.get_params(deep)
    
    # Train/Tune/Evaluate functions ---------------------------------------
    def fit(self, X, y):
        
        # Keep a reference of features utilised 
        self._features = X.columns 

        # Fit model 
        if self._tune_props == None: # Without hyperparameter tuning 
            self._model.fit(X, y)
        else:                        # Hyperparameter tuning 
            self._tuner = tu.tune_model(self._model, X, y, self._tune_props)
            self._model = self._tuner.best_estimator_
    
    def predict(self, X):
        return self._model.predict(X)
  
    def evaluate(self, metrics, X, y):
        y_pred = self.predict(X)
        return ev.evaluate(metrics, y, y_pred)
    
###### Random Forest classifier ##########################################
class RandomForestAlgo(_BaseAlgo): 

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props
        )
        self._model_name = MODEL_RF
        self._model = RandomForestClassifier(**kwargs)

###### XGBoost classifier ################################################    
class XgboostAlgo(_BaseAlgo): 

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props
        )
        self._model_name = MODEL_XGB
        self._model = xgb.XGBClassifier(**kwargs)

###### LightGBM classifier ###############################################
class LightGbmAlgo(_BaseAlgo):

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props
        )
        self._model_name = MODEL_LIGHT
        self._model = lgb.LGBMClassifier(**kwargs)
    
###### CatBoost classifier ###############################################
class CatBoostAlgo(_BaseAlgo):

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props
        )
        self._model_name = MODEL_CAT
        self._model = CatBoostClassifier(**kwargs)
    