"""
'As described in Sections 2.1 and 2.2, a trade-off exists among maximum tree depth, 
learning rate and the number of iteration in GB and XGBoost, 
which violates the independence assumption in the TPE algorithm. 
Therefore, we introduce a stepwise training process. 
First, learning rate and the number of boosts are manually determined. 
We follow the default learning rate 0.1, which is also suggested value in GB (Friedman, 2001).'

source: A boosted decision tree approach using Bayesian hyper-parameter optimization for credit scoring
url: https://www.sciencedirect.com/science/article/abs/pii/S0957417417301008

Hyperparameter search space for learning rate 
source: https://github.com/catboost/benchmarks (quality benchmarks)

- XGBoost:  ['eta': hp.loguniform('eta', -7, 0)]
- LightGBM: ['learning_rate': hp.loguniform('learning_rate', -7, 0)]
- CatBoost: ['learning_rate': hp.loguniform('learning_rate', -5, 0)]

"""

# TODO -> This code should be refactored and moved to tuning module 

import math
import optuna
import pickle
import numpy as np

import cryptoaml.datareader as cdr
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# constants 
folds = 5
train_size = 0.7
rs_iterations = 50
estimators = 5000
dataset = "eth_accounts" # elliptic, eth_accounts
feature_set = "ALL"      # elliptic [LF, LF_NE, AF, AF_NE], eth_accounts [ALL]
model = "lightgbm"       # xgboost, lightgbm, catboost 

save_file = "rs_{}_{}.pkl".format(model, feature_set)
stratify_shuffle = True
use_gpu = False
n_jobs = -1

# loads dataset 
data = cdr.get_data(dataset)
data_split = data.train_test_split(train_size=train_size)
X = data_split[feature_set].train_X
y = data_split[feature_set].train_y

# custom f1 score eval function for lgb
def lgb_f1_score(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = np.round(y_pred).astype(int)
    return ("F1", f1_score(y_true, y_pred, average="binary"), True)

# custom f1 score eval function for xgb
def xgb_f1_score(y_pred, y_true):
    y_true = y_true.get_label().astype(int)
    y_pred = [1 if y_cont > 0.5 else 0 for y_cont in y_pred] 
    return ("F1", f1_score(y_true, y_pred, average="binary"))

# objective for Random Search maximise F1 score 
def objective(trial):
    param_grid = {}
    score = None 

    # setting up learning rate 
    estimator = None 
    if model == "lightgbm":
        param_grid["learning_rate"] = trial.suggest_loguniform("learning_rate", math.exp(-7), math.exp(0))
        param_grid["n_estimators"] = estimators
        param_grid["n_jobs"] = n_jobs
        estimator = lgb.LGBMClassifier(**param_grid)
    elif model == "xgboost":
        param_grid["learning_rate"] = trial.suggest_loguniform("learning_rate", math.exp(-7), math.exp(0))
        param_grid["n_estimators"] = estimators
        param_grid["n_jobs"] = n_jobs
        estimator = xgb.XGBClassifier(**param_grid)
    elif model == "catboost":
        param_grid["learning_rate"] = trial.suggest_loguniform("learning_rate",  math.exp(-7), math.exp(0))
        param_grid["eval_metric"] = "F1"
        param_grid["bootstrap_type"] = "MVS"
        param_grid["iterations"] = estimators
        param_grid["thread_count"] = n_jobs
        if use_gpu:
            param_grid["task_type"] = "GPU"
            param_grid["devices"] = "0"

        param_grid["verbose"] = 0
        estimator = CatBoostClassifier(**param_grid)

    rs = None 
    if stratify_shuffle == True:
        rs = 42

    evals_results = []
    cross_val = StratifiedKFold(n_splits=folds, shuffle=stratify_shuffle, random_state=rs)
    for train_index, test_index in cross_val.split(X, y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]       

        fit_props = None
        eval_result_name = None 
        eval_result_metric = "F1" 

        if model == "lightgbm":
            eval_result_name = "test"
            fit_props = {
                "X":X_train,
                "y":y_train,
                "eval_names":eval_result_name,
                "eval_metric":lgb_f1_score,
                "verbose":False,
                "eval_set":[(X_test, y_test)]
            }
        elif model == "catboost":
            eval_result_name = "learn"
            fit_props = {
                "X":X_train,
                "y":y_train,
                "verbose":False,
                "eval_set":[(X_test, y_test)]
            }
        elif model == "xgboost":
            eval_result_name = "validation_0"
            fit_props = {
                "X":X_train,
                "y":y_train,
                "verbose":False,
                "eval_metric": xgb_f1_score,
                "eval_set":[(X_test, y_test)]
            }
        
        results = None
        estimator.fit(**fit_props)
        print(estimator.get_params())

        if model == "lightgbm":
            results = estimator.evals_result_[eval_result_name][eval_result_metric]
        elif model == "catboost":
            _, results, _ = np.genfromtxt('catboost_info/test_error.tsv', delimiter="\t", unpack=True, skip_header=1)           
        elif model == "xgboost":
            results = estimator.evals_result()[eval_result_name][eval_result_metric]
            print(len(results))

        evals_results.append(results)

    mean_evals_results = np.mean(evals_results, axis=0)
    best_n_estimators = np.argmax(mean_evals_results) + 1
    trial.set_user_attr("best_n_estimators", best_n_estimators)

    std_evals_results = np.std(evals_results, axis=0)
    trial.set_user_attr("cv_std", std_evals_results[best_n_estimators - 1])
    
    min_evals_results = np.min(evals_results, axis=0)
    trial.set_user_attr("cv_min", min_evals_results[best_n_estimators - 1])

    max_evals_results = np.max(evals_results, axis=0)
    trial.set_user_attr("cv_max", max_evals_results[best_n_estimators - 1])
    
    if np.isnan(mean_evals_results[best_n_estimators - 1]):
        score = 0
    else: 
        print("Best n_estimators: {}".format(best_n_estimators))
        score = mean_evals_results[best_n_estimators - 1]

    return score

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), 
                            direction="maximize")
study.set_user_attr("k_folds", folds)
study.set_user_attr("cv_method", "StratifiedKFold")
study.optimize(objective, n_trials=rs_iterations, n_jobs=1)

with open(save_file, "wb") as model_file:
    pickle.dump(study, model_file)
