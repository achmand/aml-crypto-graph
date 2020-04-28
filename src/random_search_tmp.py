import numpy as np
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb
import optuna
import pickle

from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import cryptoaml.datareader as cdr
from cryptoaml.models import XgboostAlgo
from cryptoaml.models import LightGbmAlgo
from cryptoaml.models import CatBoostAlgo
from catboost import CatBoostClassifier

# elliptic dataset
# elliptic = cdr.get_data("elliptic")
# data = elliptic.train_test_split(train_size=0.7, feat_set="LF")
# data = elliptic.train_test_split(train_size=0.7, feat_set="LF_NE")
# data = elliptic.train_test_split(train_size=0.7, feat_set="AF")
# data = elliptic.train_test_split(train_size=0.7, feat_set="AF_NE")
# print(data.train_X.shape)
# train_X = data.train_X
# train_y = data.train_y

# eth accounts dataset
eth_accounts = cdr.get_data("eth_accounts")
data = eth_accounts.train_test_split(train_size=0.7)
print(data["ALL"].train_X.shape)
train_X = data["ALL"].train_X
train_y = data["ALL"].train_y

def objective(trial):
    
    param = {
        # FOR XGB and LightBoost
        "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.05, 0.3, 0.0025)
        
        # FOR CAT BOOST 
        # "verbose": 0, 
        # "task_type": "GPU", 
        # "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.01, 0.3, 0.0025)
    }

    # FOR XGB and LightBoost
    if param["learning_rate"] < 0.1:
        param["n_estimators"] = trial.suggest_int("n_estimators", 400, 1000, 25)
    else: 
        param["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, 25)

    # FOR CATBOOST 
    # if param["learning_rate"] < 0.1:
    #     param["iterations"] = trial.suggest_int("iterations", 400, 1000, 25)
    # else: 
    #     param["iterations"] = trial.suggest_int("iterations", 100, 500, 25)


    tmp_estimator = XgboostAlgo(**param)
    # tmp_estimator = LightGbmAlgo(**param)
    # tmp_estimator = CatBoostClassifier(**param)

    scores = cross_val_score(tmp_estimator, 
                             train_X, 
                             train_y, 
                             scoring="f1", 
                             verbose=3,
                             cv=StratifiedKFold(n_splits=10),
                             n_jobs=-1)
    
    print(tmp_estimator.get_params())
    mean_score = scores.mean()
    trial.set_user_attr("cv_mean", mean_score)    
    std_score  = scores.std()
    trial.set_user_attr("cv_std", std_score)
    min_score  = scores.min()
    trial.set_user_attr("cv_min", min_score)
    max_score  = scores.max()
    trial.set_user_attr("cv_max", max_score)

    return mean_score

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), 
                            direction="maximize")

study.set_user_attr("k_folds", 10)
study.set_user_attr("cv_method", "StratifiedKFold")
study.optimize(objective, n_trials=100, n_jobs=1)

with open("rs_xgboost_ALL.pkl", "wb") as model_file:
    pickle.dump(study, model_file)
