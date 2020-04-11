import numpy as np
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb
import optuna
# save study
import pickle


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import cryptoaml.datareader as cdr
from cryptoaml.models import XgboostAlgo

elliptic = cdr.get_data("elliptic")
data = elliptic.train_test_split(train_size=0.7, feat_set="LF")

print(data.train_X.shape)

train_X = data.train_X
train_y = data.train_y

test_X = data.test_X
test_y = data.test_y

tmp_estimator = XgboostAlgo()
def objective(trial):
    
    param = {
        # using RS
        "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.05, 0.3, 0.025),
#         "n_estimators": trial.suggest_int("n_estimators", 100, 800, 25),
        
        
         "tree_method":"gpu_hist", 
         "predictor":"gpu_predictor"
        
#         "max_depth": trial.suggest_int("max_depth", 1, 12),
#         "subsample": trial.suggest_uniform("subsample", 0.9, 1.0),
#         "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.9, 1.0),       
#         "min_child_weight": trial.suggest_int("min_child_weight", 0, 4),
#         "max_delta_step": trial.suggest_int("max_delta_step", 0, 1),
#         "gamma": trial.suggest_loguniform("gamma", 0.0, 0.01),       
#         "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
#         "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1.0)
    }

    if param["learning_rate"] < 0.1:
        param["n_estimators"] = trial.suggest_int("n_estimators", 400, 1000, 25)
    else: 
        param["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, 25)

    
    tmp_estimator.set_params(**param)
    scores = cross_val_score(tmp_estimator, 
                             train_X, 
                                 train_y, 
                                 scoring="neg_log_loss", 
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

# study = optuna.create_study(sampler=optuna.samplers.TPESampler(), 
#                             direction="maximize")



study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), 
                            direction="maximize")


# study = optuna.create_study(sampler=optuna.samplers.GridSampler({
#     "n_estimators": [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]}), 
#                             direction="maximize")

study.set_user_attr("k_folds", 3)
study.set_user_attr("cv_method", "StratifiedKFold")
study.optimize(objective, n_trials=100, n_jobs=1)

with open("gs_xgboost_LF.pkl", "wb") as model_file:
    pickle.dump(study, model_file)
