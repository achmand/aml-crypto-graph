"""
Boosting classifiers tested in the experiments.
The following models are included; 
- Extreme Gradient Boosting (XGBoost)
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import pickle
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from .. import tune as tu 
from .. import metrics as ev 
from abc import ABC, abstractmethod

###### Base classifier  ###################################################
class _BaseAlgo(ABC):
    def __init__(self, **kwargs):
        self._model = None
        self._init_model(**kwargs)

    @property
    def params(self):
        return self._model.get_params()

    @property
    def feature_importance(self):
        
        # return a pandas dataframe with feature importance 
        return pd.DataFrame(self._model.feature_importances_,
                            index = self._column_names,
                            columns=["importance"]).sort_values("importance", ascending=False)

    @abstractmethod 
    def _init_model(self, **kwargs):
        pass
    
    def save(self):
        print("Saving model")

    def fit(self, X_train, y_train, tune=None):
        
        # Keep a reference to the column names
        # -> To be able to output the feature importances  
        self._column_names = X_train.columns

        # No hyperparameter tuning 
        if tune == None:
            self._model.fit(X_train, y_train)
        else: 
            self._tune = tune 
            self._tune_method = tu.tune_model(self._model, X_train, y_train, self._tune)
            self._model = self._tune_method.best_estimator_ 

    def predict(self, X_test):
        return self._model.predict(X_test)

    def evaluate(self, metrics, X_test, y_test):
        y_pred = self.predict(X_test)
        return ev.evaluate(metrics, y_test, y_pred)

###### XGBoost classifier  ################################################    
class XgbBoostAlgo(_BaseAlgo):
    
    def __init__(self, **kwargs):

        # Call base constructor 
        super().__init__(**kwargs)

    def _init_model(self, **kwargs):
        
        # New instance of xgboost classifier with the specified args
        self._model = xgb.XGBClassifier(**kwargs)

###### LightGBM classifier  ###############################################
# class LightGbmAlgo(_BaseAlgo):

#     def __init__(self, **kwargs):

#         # New instance of lightGBM classifier with the specified args
#         self._model = lgb.LGBMClassifier(kwargs)
    