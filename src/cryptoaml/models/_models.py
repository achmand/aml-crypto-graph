"""

"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import json
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
# TODO -> Implement logger 

###### Constants #########################################################
MODEL_BASE   = "model_base"
MODEL_RF     = "random_forest"
MODEL_XGB    = "xg_boost"
MODEL_XGB_RF = "xg_boost_rf"
MODEL_LIGHT  = "light_boost"
MODEL_CAT    = "cat_boost"

PERSIST_SAVE = "save"
PERSIST_LOAD = "load"

###### base classifier ###################################################
class _BaseAlgo(ABC, BaseEstimator, ClassifierMixin):
    """
    Base class for models.
    Warning: This class should not be used directly. Use derived classes instead.
    """

    # constructor ---------------------------------------------------------
    def __init__(self, 
                 tune_props=None, 
                 persist_props=None,
                 **kwargs):
        self._model = None 
        self._model_name = MODEL_BASE
        self._tuner = None 
        self._tune_props = tune_props
        self._init_model(**kwargs)

        # convert persistence properties to Namespace
        self._persist_props = persist_props
        if persist_props != None:
            self._persist_props = u.Namespace(persist_props)

            # load model
            if self._persist_props.method == PERSIST_LOAD:
                self.load(self._persist_props.load_path)                       
            
    # properties ----------------------------------------------------------
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
    def tune_props_(self):
        if self._tune_props is None:
            raise TypeError("'tune_props not passed'")
        return self._tune_props

    @property
    def tune_results_(self):
        if self._tune_props is None:
            raise TypeError("'tune_props' not passed'")
        check_is_fitted(self._model, ["feature_importances_"])
        return (self._tuner.meta_results_, self._tuner.results_)
        
    # init/parameters functions -------------------------------------------
    @abstractmethod 
    def _init_model(self, **kwargs):
        pass

    def set_params(self, **params):
        return self._model.set_params(**params)
    
    def get_params(self, deep=True):
        return self._model.get_params(deep)

    # train/tune/evaluate functions ---------------------------------------
    # TODO-> Add the ability to not tune even if the tune_props are passed 
    def fit(self, X, y, tune=True):
        
        self._model.set_params({n_jobs=16})

        # Keep a reference of features 
        self._features = X.columns.values

        # Fit model 
        # No hyperparameter tuning 
        if self._tune_props == None or tune==False: 
            self._model.fit(X, y)
        # Hyperparameter tuning 
        else:                        
            self._tuner, self._tune_props = tu.tune_model(self._model, X, y, self._tune_props)
            self._model = self._tuner.best_estimator_
        
        # Persist model after training 
        if self._persist_props != None and self._persist_props.method == PERSIST_SAVE:
            print("Saving model")
            self.save(self._persist_props.save_path)

    def predict(self, X):
        return self._model.predict(X)
    
    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def evaluate(self, metrics, X, y):
        y_pred = self.predict(X)
        return ev.evaluate(metrics, y, y_pred)

    # persistence functions -----------------------------------------------
    def save(self, path):
        
        # Make sure model is fitted before saving 
        check_is_fitted(self._model, ["feature_importances_"])

        # Save model 
        source_path = path + "/" + self._model_name
        u.create_dir(source_path)
        with open(source_path + "/" + self._model_name + ".pkl", "wb") as model_file:
            pickle.dump(self._model, model_file)

        # Save meta data
        meta_data = {}
        meta_data["features"] = self._features
        meta_data["tune_props"] = self._tune_props
        with open(source_path + "/" + self._model_name + "_meta.pkl", "wb") as meta_file:
            pickle.dump(meta_data, meta_file, pickle.HIGHEST_PROTOCOL)

        # Save tuner if set 
        if self._tune_props != None: 
            with open(source_path + "/" + self._model_name + "_tuner.pkl", "wb") as tuner_file:
                pickle.dump(self._tuner, tuner_file)

    def load(self, path):
        
        # loads model
        with open(path + "/" +  path.split("/")[-1] + ".pkl", "rb") as model_file:
            tmp_model = pickle.load(model_file)
            if type(tmp_model) != type(self._model):
                raise TypeError("model loaded is of type '{}' but expecting '{}'".format(type(tmp_model), type(self._model)))
            self._model = tmp_model

        # Load meta data 
        with open(path + "/" +  path.split("/")[-1] + "_meta.pkl", "rb") as meta_file:
            meta_data = pickle.load(meta_file)
            self._features = meta_data["features"]
            self._tune_props = meta_data["tune_props"] 

        # load tuner if set 
        if self._tune_props != None: 

            # fix to solve issue with DEAP import when unpickling tuner 
            method = self._tune_props["method"]
            del self._tune_props["method"]
            self._tune_props["estimator"] = self._model
            self._tuner = tu.get_tuner(method, **self._tune_props)

            with open(path + "/" +  path.split("/")[-1] + "_tuner.pkl", "rb") as tuner_file:
                self._tuner = pickle.load(tuner_file)

###### Random Forest classifier ##########################################
class RandomForestAlgo(_BaseAlgo): 

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 persist_props=None,
                 **kwargs): 
        super().__init__(
            tune_props=tune_props, 
            persist_props=persist_props,
            **kwargs
        )
        
    # Init/Parameters functions -------------------------------------------
    def _init_model(self, **kwargs):
        self._model_name = MODEL_RF
        self._model = RandomForestClassifier(**kwargs)

###### XGBoost classifier ################################################    
class XgboostAlgo(_BaseAlgo): 

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 persist_props=None,
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props,
            persist_props = persist_props,
            **kwargs
        )

    # Init/Parameters functions -------------------------------------------
    def _init_model(self, **kwargs):
        self._model_name = MODEL_XGB
        self._model = xgb.XGBClassifier(**kwargs)

###### XGBoostRf classifier ##############################################    
class XgboostRfAlgo(_BaseAlgo): 

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 persist_props=None,
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props,
            persist_props = persist_props,
            **kwargs
        )

    # Init/Parameters functions -------------------------------------------
    def _init_model(self, **kwargs):
        self._model_name = MODEL_XGB_RF
        self._model = xgb.XGBRFClassifier(**kwargs)

###### LightGBM classifier ###############################################
class LightGbmAlgo(_BaseAlgo):

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 persist_props=None,
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props,
            persist_props = persist_props,
            **kwargs
        )

    # Init/Parameters functions -------------------------------------------
    def _init_model(self, **kwargs):   
        self._model_name = MODEL_LIGHT
        self._model = lgb.LGBMClassifier(**kwargs)
    
###### CatBoost classifier ###############################################
class CatBoostAlgo(_BaseAlgo):

    # Constructor ---------------------------------------------------------
    def __init__ (self, 
                 tune_props=None, 
                 persist_props=None,
                 **kwargs): 
        super ().__init__(
            tune_props = tune_props,
            persist_props = persist_props,
            **kwargs
        )
    
    # Init/Parameters functions -------------------------------------------
    def _init_model(self, **kwargs):
        self._model_name = MODEL_CAT
        self._model = CatBoostClassifier(**kwargs)

    # init/parameters functions -------------------------------------------
    def get_params(self):
        return self._model.get_all_params()