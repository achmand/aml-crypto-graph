"""
Boosting classifiers tested in the experiments.
The following models are included; 
- Extreme Gradient Boosting (XGBoost)
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

# Boosting Models 
import xgboost as xgb
import lightgbm as lgb
from logitboost import LogitBoost
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

###### Base classifier  ###################################################
class _BaseAlgo(ABC):

    # Constructor ---------------------------------------------------------
    def __init__(self, **kwargs):
        self._model = None
        self._init_model(**kwargs)

    # Properties ----------------------------------------------------------
    @property
    def params(self):
        return self._model.get_params()

    @property
    def feature_importance(self):
        
        # Return a pandas dataframe with feature importance 
        return pd.DataFrame(self._model.feature_importances_,
                            index = self._column_names,
                            columns=["importance"]).sort_values("importance", ascending=False)

    @abstractmethod 
    def _init_model(self, **kwargs):
        pass
    
    # Train/Tune/Evaluate functions ---------------------------------------
    def fit(self, X_train, y_train, tune_props=None):
        
        # Keep a reference to the column names
        # -> To be able to output the feature importances  
        self._column_names = X_train.columns

        # No hyperparameter tuning 
        if tune_props == None:
            self._model.fit(X_train, y_train)
        else: 
            self._tune_props = tune_props  # TODO must be part of tune method 
            self._tune_method = tu.tune_model(self._model, X_train, y_train, self._tune_props)
            self._model = self._tune_method.best_estimator_ 

    def predict(self, X_test):
        return self._model.predict(X_test)

    def evaluate(self, metrics, X_test, y_test):
        y_pred = self.predict(X_test)
        return ev.evaluate(metrics, y_test, y_pred)
    
    # Persistence functions -----------------------------------------------
    def save(self, path):

        # Get model name from path 
        model_name = path.split("/")[-1]

        # Saves model in specified location 
        u.create_dir(path)

        # Saves model
        model_file = path + "/" + model_name 
        pickle.dump(self._model, open(model_file + ".pkl", "wb"))

        # Saves column names 
        np.savetxt(model_file + ".cols", self._column_names, delimiter=",", fmt="%s")
    
    def load(self, path):

        # Get model name from path 
        model_name = path.split("/")[-1]
        model_file = path + "/" + model_name  

        # Load model 
        self._model = pickle.load(open(model_file+ ".pkl", "rb"))

        # Load column names 
        self._column_names = np.loadtxt(model_file + ".cols", delimiter=",", dtype="str")

###### AdaBoost classifier  ###############################################
class AdaBoostAlgo(_BaseAlgo):
    
    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of AdaBoost classifier with the specified args
        self._model = AdaBoostClassifier(**kwargs)

###### LogitBoost classifier  #############################################
class LogitBoostAlgo(_BaseAlgo):

    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of LogitBoost classifier with the specified args
        self._model = LogitBoost(**kwargs)

###### Gradient Boosting classifier  ######################################
class GradientBoostAlgo(_BaseAlgo):
    
    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of Gradient Boosting classifier with the specified args
        self._model = GradientBoostingClassifier(**kwargs)

###### XGBoost classifier  ################################################    
class XgbBoostAlgo(_BaseAlgo): 
    
    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of Xgboost classifier with the specified args
        self._model = xgb.XGBClassifier(**kwargs)

###### LightGBM classifier  ###############################################
class LightGbmAlgo(_BaseAlgo):
    
    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of LightGBM classifier with the specified args
        self._model = lgb.LGBMClassifier(**kwargs)

###### CatBoost classifier  ###############################################
class CatBoostAlgo(_BaseAlgo):
    
    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of CatBoost classifier with the specified args
        self._model = CatBoostClassifier(**kwargs)