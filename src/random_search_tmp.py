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

elliptic = cdr.get_data("elliptic")
data = elliptic.train_test_split(train_size=0.7, feat_set="AF_NE")

print(data.train_X.shape)

train_X = data.train_X
train_y = data.train_y
test_X = data.test_X
test_y = data.test_y

scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True, eps=1e-7)
tmp_estimator = XgboostAlgo()
def objective(trial):
    
    param = {
        "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.05, 0.3, 0.025),
        "tree_method":"gpu_hist", 
        "predictor":"gpu_predictor"
    }

    if param["learning_rate"] < 0.1:
        param["n_estimators"] = trial.suggest_int("n_estimators", 400, 1000, 25)
    else: 
        param["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, 25)

    
    tmp_estimator.set_params(**param)
    scores = cross_val_score(tmp_estimator, 
                             train_X, 
                             train_y, 
                             scoring=scorer, 
                             cv=StratifiedKFold(n_splits=3),
                             n_jobs=1)
    
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

study.set_user_attr("k_folds", 3)
study.set_user_attr("cv_method", "StratifiedKFold")
study.optimize(objective, n_trials=100, n_jobs=1)

with open("gs_xgboost_AF_NE.pkl", "wb") as model_file:
    pickle.dump(study, model_file)
