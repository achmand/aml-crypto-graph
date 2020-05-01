"""
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies ############################################
import random
import numpy as np
import pandas as pd
from .. import utils as u 
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score

import optuna
from hyperopt import fmin, tpe, atpe, rand, space_eval, Trials, STATUS_OK

from sklearn.metrics import get_scorer
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

###### constants #########################################################
TUNE_BASE                = "tuner_base"           # Base tuner 
TUNE_TPE                 = "tpe"                  # Tree of Parzen Estimators 
TUNE_ATPE                = "atpe"                 # Adaptive Tree of Parzen Estimators 
TUNE_RANDOM_SEARCH       = "random_search"        # Random Search 
TUNE_EVOLUTIONARY_SEARCH = "evolutionary_search"  # Genetic Algorithm 
TUNE_CMA_ES              = "cma_es"               # Covariance Matrix Adaptation Evolution Strategy

# TODO -> validation where needed 

###### Base Tuner ########################################################
class _BaseTuner(ABC):
    """
    Base class for hyper parameter tuners for models.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """ 

    # Constructor ---------------------------------------------------------
    def __init__(self, 
                 estimator, 
                 param_grid, 
                 scoring,
                 k_folds, 
                 verbose=1):
        self._tuner = None
        self._tuner_name = TUNE_BASE
        self._estimator = estimator 
        self._param_grid = param_grid
        self._scoring = scoring
        self._k_folds = k_folds
        self._verbose = verbose
    
    # Properties ----------------------------------------------------------
    @property
    @abstractmethod
    def best_estimator_(self):
        pass

    @property
    @abstractmethod
    def best_score_(self):
        pass

    @property
    @abstractmethod
    def best_params_(self):
        pass

    @property
    @abstractmethod
    def scorer_(self):
        pass

    @property
    def meta_results_(self):
        meta = {
            "scorer": self.scorer_,
            "best_score": self.best_score_,
            "best_params": self.best_params_
        }

        return meta

    @property
    @abstractmethod
    def results_(self):
        results = pd.DataFrame(self._tuner.cv_results_).sort_values(
            "mean_test_score", 
            ascending=False)

        return results

    # Train/Tune functions -----------------------------------------------
    @abstractmethod 
    def fit(self, X, y):
        pass 

###### Optina tuner: TPE #################################################
class OptunaTuner(_BaseTuner):

    # Constructor ---------------------------------------------------------
    def __init__(self,
                 estimator,
                 param_grid,
                 scoring, 
                 k_folds, 
                 n_iterations,
                 stratify_shuffle=True,
                 verbose=1):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid, 
            scoring=scoring,
            k_folds=k_folds,
            verbose=verbose
        )

        self._n_iterations = n_iterations
        self._estimator_class = type(estimator)
        self._stratify_shuffle = stratify_shuffle
        # self._scorer = get_scorer("neg_log_loss")
        # to fix divide by zero problem in log loss 
        # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48701
        self._scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True, eps=1e-7)
        self._sampler = optuna.samplers.TPESampler()
        self._study = optuna.create_study(sampler=self._sampler, direction="maximize")
        self._study.set_user_attr("k_folds", k_folds)
        self._study.set_user_attr("cv_method", "StratifiedKFold")

        # get reference to 'throw' and 'fixed' parameters
        # => 'throw': only use in tuning process 
        # => 'fixed': values is fixed 
        # => any other will be used in search space 
        throw_prop = {}
        fixed_prop = {}
        search_space = {}
        
        for prop_key in self._param_grid:
            tmp_property = self._param_grid[prop_key]
            tmp_property_type = tmp_property["type"]
            if tmp_property_type == "throw":
                throw_prop[prop_key] = tmp_property["value"]
            elif tmp_property_type == "fixed":
                fixed_prop[prop_key] = tmp_property["value"]
            else:
                search_space[prop_key] = tmp_property 

        self._throw_prop = throw_prop
        self._fixed_prop = fixed_prop
        self._search_space = search_space

    # Properties ----------------------------------------------------------
    @property
    def best_estimator_(self):
        return self._best_estimator

    @property
    def best_score_(self):
        return self._best_score

    @property
    def best_params_(self):
        return self._best_params

    @property
    def scorer_(self):
        return self._scorer

    @property
    def results_(self):
        return self._results.sort_values(
            "value", 
            ascending=False)

    # train/tune functions -----------------------------------------------
    def _new_params(self, trial): 
        
        # get new parameters to evaluate 
        new_params = {}

        # 1. set 'throw' parameters 
        for throw_key in self._throw_prop:
            new_params[throw_key] = self._throw_prop[throw_key]
        
        # 2. set 'fixed' parameters 
        for fixed_key in self._fixed_prop:
            new_params[fixed_key] = self._fixed_prop[fixed_key]

        # 3. set 'search' parameters
        for search_key in self._search_space:
            tmp_prop = self._search_space[search_key]
            tmp_prop_type = tmp_prop["type"]
            if tmp_prop_type == "suggest_categorical":
                tmp_prop_options = tmp_prop["choices"]
                new_params[search_key] =  trial.suggest_categorical(search_key,tmp_prop_options)
                continue
            
            tmp_prop_min = tmp_prop["min"]
            tmp_prop_max = tmp_prop["max"]
            if tmp_prop_type == "suggest_int":
                step = tmp_prop.get("step", 1)
                new_params[search_key] = trial.suggest_int(search_key, tmp_prop_min, tmp_prop_max, step)
            elif tmp_prop_type == "suggest_uniform":
                new_params[search_key] = trial.suggest_uniform(search_key, tmp_prop_min, tmp_prop_max)
            elif tmp_prop_type == "suggest_discrete_uniform":
                step = tmp_prop.get("step", 0.005)
                new_params[search_key] = trial.suggest_discrete_uniform(search_key, tmp_prop_min, tmp_prop_max, step)
            elif tmp_prop_type == "suggest_loguniform":
                new_params[search_key] = trial.suggest_loguniform(search_key, tmp_prop_min, tmp_prop_max)
            else:
                raise NotImplementedError("The type '{}' for parameter is not yet implemented".format(tmp_prop_type))

        return new_params    

    # custom f1 score eval function for lgb
    def _lgb_f1_score(self, y_true, y_pred):
        y_true = y_true.astype(int)
        y_pred = np.round(y_pred).astype(int)
        return ("F1", f1_score(y_true, y_pred, average="binary"), True)

    # custom f1 score eval function for xgb
    def _xgb_f1_score(self, y_pred, y_true):
        y_true = y_true.get_label().astype(int)
        y_pred = [1 if y_cont > 0.5 else 0 for y_cont in y_pred] 
        return ("F1", f1_score(y_true, y_pred, average="binary"))

    # for other models 
    def _objective_normal(self, trial):
        
        params = self._new_params(trial)
        tmp_estimator = self._estimator_class(**params)
        scores = cross_val_score(tmp_estimator, 
                                 self._X, 
                                 self._y, 
                                 verbose=3,
                                 scoring="f1", 
                                 n_jobs=-1,
                                 cv=StratifiedKFold(n_splits=self._k_folds, shuffle=self._stratify_shuffle))
   
        mean_score = scores.mean()
        trial.set_user_attr("cv_mean", mean_score)    
        std_score  = scores.std()
        trial.set_user_attr("cv_std", std_score)
        min_score  = scores.min()
        trial.set_user_attr("cv_min", min_score)
        max_score  = scores.max()
        trial.set_user_attr("cv_max", max_score)

        return mean_score

    # for boosting models (includes exhaustive search on number of boosters)
    def _objective(self, trial):
        
        params = self._new_params(trial)
        tmp_estimator = self._estimator_class(**params)

        rs = None 
        if self._stratify_shuffle == True:
            rs = 42

        score = None
        evals_results = []
        cross_val = StratifiedKFold(n_splits=self._k_folds, shuffle=self._stratify_shuffle, random_state=rs)
        for train_index, test_index in cross_val.split(self._X, self._y):
            X_train, y_train = self._X.iloc[train_index], self._y.iloc[train_index]
            X_test, y_test = self._X.iloc[test_index], self._y.iloc[test_index]       

            fit_props = None
            eval_result_name = None 
            eval_result_metric = "F1" 

            if isinstance(tmp_estimator, lgb.LGBMClassifier):
                eval_result_name = "test"
                fit_props = {
                    "X":X_train,
                    "y":y_train,
                    "eval_names":eval_result_name,
                    "eval_metric":self._lgb_f1_score,
                    "verbose":False,
                    "eval_set":[(X_test, y_test)]
                }
            elif isinstance(tmp_estimator, CatBoostClassifier):
                eval_result_name = "learn"
                fit_props = {
                    "X":X_train,
                    "y":y_train,
                    "verbose":False,
                    "eval_set":[(X_test, y_test)]
                }
            elif isinstance(tmp_estimator, xgb.XGBClassifier):
                eval_result_name = "validation_0"
                fit_props = {
                    "X":X_train,
                    "y":y_train,
                    "verbose":False,
                    "eval_metric": self._xgb_f1_score,
                    "eval_set":[(X_test, y_test)]
                }

            results = None
            tmp_estimator.fit(**fit_props)

            if isinstance(tmp_estimator, lgb.LGBMClassifier):
                print("extract stats from LightGBM eval")
                results = tmp_estimator.evals_result_[eval_result_name][eval_result_metric]
            elif isinstance(tmp_estimator, CatBoostClassifier):
                print("extract stats from CatBoost eval")
                _, results, _ = np.genfromtxt('catboost_info/test_error.tsv', delimiter="\t", unpack=True, skip_header=1)           
            elif isinstance(tmp_estimator, xgb.XGBClassifier):
                print("extract stats from XGBoost eval")
                results = tmp_estimator.evals_result()[eval_result_name][eval_result_metric]

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
    
    def fit(self, X, y):

        self._X = X
        self._y = y

        is_boosting = self._estimator_class == lgb.LGBMClassifier or self._estimator_class == CatBoostClassifier or self._estimator_class == xgb.XGBClassifier
        if is_boosting:
            self._study.optimize(self._objective, n_trials=self._n_iterations) 
        else:
            self._study.optimize(self._objective_normal, n_trials=self._n_iterations) 

        # gets and sets best score from trials
        self._best_score = self._study.best_trial.value

        # set results from trials
        self._results = self._study.trials_dataframe() 

        # train a model with best params and set as best estimator
        # 1. get best parameters found
        params = self._study.best_trial.params  

        # 2. concat best parameters with fixed ones as they are not returned     
        if self._estimator_class == lgb.LGBMClassifier:
            print("Setting best n_estimator for LGBM")
            best_n_estimators = self._study.best_trial.user_attrs["best_n_estimators"]
            params["n_estimators"] = best_n_estimators
        elif self._estimator_class == CatBoostClassifier:
            print("Setting best n_estimator for CatBoost")
            best_n_estimators = self._study.best_trial.user_attrs["best_n_estimators"]
            params["iterations"] = best_n_estimators
        elif self._estimator_class == xgb.XGBClassifier:
            print("Setting best n_estimator for XGBoost")
            best_n_estimators = self._study.best_trial.user_attrs["best_n_estimators"]
            params["n_estimators"] = best_n_estimators

        best_params = {**self._fixed_prop, **params}       
        self._best_params = best_params

        print("Best parameters: {}".format(best_params))
        
        # 3. train model with best parameters found  
        estimator = self._estimator_class(**best_params)
        estimator.fit(self._X, self._y)
        self._best_estimator = estimator

###### Hyperopt tuner: TPE, ATPE, Random Search ##########################
class HyperOptTuner(_BaseTuner):
    
    _algo_options = {
        TUNE_TPE:           tpe.suggest,   # Tree of Parzen Estimators 
        TUNE_ATPE:          atpe.suggest,  # Adaptive Tree of Parzen Estimators 
        TUNE_RANDOM_SEARCH: rand.suggest   # Random Search 
    }

    # Constructor ---------------------------------------------------------
    def __init__(self,
                 estimator,
                 param_grid,
                 scoring, 
                 k_folds, 
                 algo,
                 n_iterations,
                 verbose=1):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid, 
            scoring=scoring,
            k_folds=k_folds,
            verbose=verbose
        )

        # Setup algorithm for hyper opt.
        if algo not in self._algo_options:
            error = "'algo'=%r is not implemented" % algo
            raise NotImplementedError(error)
        else:
            self._algo = self._algo_options[algo]
        
        self._n_iterations = n_iterations
        self._estimator_class = type(estimator)
        self._scorer = get_scorer(scoring)

    # Properties ----------------------------------------------------------
    @property
    def best_estimator_(self):
        return self._best_estimator

    @property
    def best_score_(self):
        return self._best_score

    @property
    def best_params_(self):
        return self._best_params

    @property
    def scorer_(self):
        return self._scorer

    @property
    def results_(self):
        return self._results.sort_values(
            "score", 
            ascending=False)

    # Train/Tune functions -----------------------------------------------
    def _objective(self, params):
        
        # TODO -> Do not allow every scoring passed 

        # Create a temporary instance of the estimator and apply cv.
        tmp_estimator = self._estimator_class(**params)
        scores = cross_val_score(tmp_estimator, 
                                 self._X, 
                                 self._y, 
                                 scoring=self._scorer, 
                                 cv=StratifiedKFold(n_splits=self._k_folds))
        
        # Get statistics from scores. 
        mean_score = scores.mean()
        std_score  = scores.std()
        min_score  = scores.min()
        max_score  = scores.max()

        # Compute loss. 
        loss = 1 - mean_score

        if self._verbose == 1:
            print("iteration: {0} | loss: {1:.2f} | avg: {2:.2f} | min: {3:.2f} | max: {4:.2f} | std: {5:.2f}".format(
                self._current_iteration, loss, mean_score, min_score, max_score, std_score))

        self._current_iteration += 1
        return {
            "loss": loss,
            "params": params,
            "score": mean_score,
            "std_score": std_score, 
            "min_score": min_score,
            "max_score": max_score,
            "status": STATUS_OK
        } 
        
    def fit(self, X, y):
        self._X = X
        self._y = y

        trials = Trials()
        self._current_iteration = 0
        best_params = fmin(fn=self._objective,            # Objective function to minimize 
                           space=self._param_grid,       # Parameter grid
                           algo=self._algo,              # Type of algorithm used for search
                           max_evals=self._n_iterations, # Number of iterations 
                           trials=trials)                # Keeps record of each trail/iteration
        
        # Get best parameters and re-train model. 
        params = space_eval(self._param_grid, best_params)
        self._best_params = params

        # gets and sets best score from trials
        self._best_score = max([x["score"] for x in trials.results])

        # Set results from trials
        self._results = pd.DataFrame(trials.results) 

        # Train a model with best params and set as best estimator.
        estimator = self._estimator_class(**params)
        estimator.fit(self._X, self._y)
        self._best_estimator = estimator

###### Evolutionary Search ###############################################
class EvolutionarySearchTuner(_BaseTuner):

    # Constructor ---------------------------------------------------------
    def __init__(self, 
                 estimator, 
                 param_grid, 
                 scoring,
                 k_folds, 
                 population_size,
                 gene_mutation_prob,
                 gene_crossover_prob,
                 tournament_size,
                 generations_number,
                 n_jobs,
                 verbose=1):
        super().__init__(
            estimator=estimator,
            param_grid=param_grid, 
            scoring=scoring,
            k_folds=k_folds,
            verbose=verbose
        )
        self._tuner_name = TUNE_EVOLUTIONARY_SEARCH
        self._tuner = EvolutionaryAlgorithmSearchCV(estimator=estimator,
                                                    params=param_grid,
                                                    scoring=scoring,
                                                    cv=StratifiedKFold(n_splits=k_folds),
                                                    verbose=verbose,
                                                    population_size=population_size,
                                                    gene_mutation_prob=gene_mutation_prob,
                                                    gene_crossover_prob=gene_crossover_prob,
                                                    tournament_size=tournament_size,
                                                    generations_number=generations_number,
                                                    n_jobs=n_jobs,
                                                    refit=True)

    # Properties ----------------------------------------------------------
    @property
    def best_estimator_(self):
        return self._tuner.best_estimator_

    @property
    def best_score_(self):
        return self._tuner.best_score_

    @property
    def best_params_(self):
        return self._tuner.best_params_

    @property
    def scorer_(self):
        return self._tuner.scorer_

    @property
    def results_(self):
        results = pd.DataFrame(self._tuner.cv_results_).sort_values(
            "mean_test_score", 
            ascending=False)

        return results[["params", "mean_test_score", "std_test_score", "min_test_score", "max_test_score"]]

    # Train/Tune functions -----------------------------------------------
    def fit(self, X, y):
        self._tuner.fit(X, y)

###### Hyperparameter tuning functions ###################################
# TODO -> not in should be changed to dictionary.get(key, default_value)
def tune_model(estimator, X, y, tune_props):

    # Validate tune arguments.
    if "param_grid" not in tune_props:
        raise ValueError("'param_grid' dictionary must be passed")

    tuner = None
    properties = u.Namespace(tune_props)

    # Set common default properties if not passed.
    if "k_folds" not in tune_props:
        properties.k_folds = 5
    if "scoring" not in tune_props: 
        properties.scoring = "f1"
    if "verbose" not in tune_props: 
        properties.verbose = 1

    # Evolutionary Search (Genetic Algorithm)
    if properties.method == TUNE_EVOLUTIONARY_SEARCH:

        # Set defaults if not passed, taken from example: https://github.com/rsteca/sklearn-deap . 
        if "n_jobs" not in tune_props: 
            properties.n_jobs = 1
        if "population_size" not in tune_props:
            properties.population_size = 50
        if "gene_mutation_prob" not in tune_props:
            properties.gene_mutation_prob = 0.10
        if "gene_crossover_prob" not in tune_props:
            properties.gene_crossover_prob = 0.5
        if "tournament_size" not in tune_props:
            properties.tournament_size = 3
        if "generations_number" not in tune_props:
            properties.generations_number = 3
                          
        # initialize evolutionary search
        tuner = EvolutionarySearchTuner(estimator=estimator,
                                        param_grid=properties.param_grid,
                                        scoring=properties.scoring, 
                                        k_folds=properties.k_folds,
                                        population_size=properties.population_size,
                                        gene_mutation_prob=properties.gene_mutation_prob,
                                        gene_crossover_prob=properties.gene_crossover_prob,
                                        tournament_size=properties.tournament_size, 
                                        generations_number=properties.generations_number,
                                        n_jobs=properties.n_jobs,
                                        verbose=properties.verbose)        
    
    # HyperOpt (Adaptive Tree of Parzen Estimators, Random Search)
    elif properties.method == TUNE_ATPE or properties.method == TUNE_RANDOM_SEARCH:
        
        # Set default if not passed. 
        if "n_iterations" not in tune_props: 
            properties.n_iterations = 1000

        # initialize hyperopt tuner
        tuner = HyperOptTuner(estimator=estimator,
                              param_grid=properties.param_grid,
                              scoring=properties.scoring,
                              k_folds=properties.k_folds,
                              algo=properties.method, 
                              n_iterations=properties.n_iterations,
                              verbose=properties.verbose)
    
    # Optuna (Tree of Parzen Estimators)
    elif properties.method == TUNE_TPE:

        # set default if not passed. 
        if "n_iterations" not in tune_props: 
            properties.n_iterations = 100
        if "stratify_shuffle" not in tune_props: 
            properties.stratify_shuffle = True

        # initialize optuna tuner
        tuner = OptunaTuner(estimator=estimator,
                            param_grid=properties.param_grid,
                            scoring="neg_log_loss",
                            k_folds=properties.k_folds,
                            n_iterations=properties.n_iterations,
                            stratify_shuffle=properties.stratify_shuffle)

    else:
        raise NotImplementedError("The specified tuning method '{}' is not yet implemented".format(properties.method))

    # tune hyperparameters
    tuner.fit(X, y)
    return tuner, properties.__dict__            

def get_tuner(method, **kwargs):
    if method == TUNE_EVOLUTIONARY_SEARCH:
        return EvolutionarySearchTuner(**kwargs)
    elif method == TUNE_TPE:
        return OptunaTuner(**kwargs)
    else:
        raise NotImplementedError("The specified tuning method '{}' is not yet implemented".format(method))
