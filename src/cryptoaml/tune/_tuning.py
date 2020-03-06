"""
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### Importing dependencies #############################################
import pandas as pd
from .. import utils as u 
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV

###### Constants #########################################################
TUNER_BASE               = "tuner_base"
TUNE_RANDOM_GRID_SEARCH  = "random_grid_search"
TUNE_EVOLUTIONARY_SEARCH = "evolutionary_search"

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
        self._tuner_name = TUNER_BASE
        self._estimator = estimator 
        self._param_grid = param_grid
        self._scoring = scoring
        self._k_folds = k_folds
        self._verbose = verbose
    
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
    def meta_results_(self):
        meta = {
            "scorer": self.scorer_,
            "best_score": self.best_score_,
            "best_params": self.best_params_
        }

        return meta

    @property
    def results_(self):
        results = pd.DataFrame(self._tuner.cv_results_).sort_values(
            "mean_test_score", 
            ascending=False)

        return results

    # Train/Tune functions -----------------------------------------------
    def fit(self, X, y):
        self._tuner.fit(X, y)

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

###### Hyperparameter tuning functions ###################################
def tune_model(estimator, X, y, tune_props):

    # Validate tune arguments 
    if "param_grid" not in tune_props:
        raise ValueError("'param_grid' dictionary must be passed")

    tuner = None
    properties = u.Namespace(tune_props)

    # Set common default properties if not passed
    if "k_folds" not in tune_props:
        properties.k_folds = 5
    if "scoring" not in tune_props: 
        properties.scoring = "f1"
    if "n_jobs" not in tune_props: 
        properties.n_jobs = 1
    if "verbose" not in tune_props: 
        properties.verbose = 1

    # Evolutionary Search 
    if properties.method == TUNE_EVOLUTIONARY_SEARCH:

        # Set defaults if not passed 
        # Taken from example: https://github.com/rsteca/sklearn-deap
        if "population_size" not in tune_props:
            properties.population_size = 50
        if "gene_mutation_prob" not in tune_props:
            properties.gene_mutation_prob = 0.10
        if "gene_crossover_prob" not in tune_props:
            properties.gene_crossover_prob = 0.5
        if "tournament_size" not in tune_props:
            properties.tournament_size = 3
        if "generations_number" not in tune_props:
            properties.generations_numbers = 3
                          
        # Initialize evolutionary search
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

        # Random Grid Search 
        # TODO -> Not yet implemented 
        if properties.method == TUNE_RANDOM_GRID_SEARCH:
            raise NotImplementedError("The specified tuning method '{}' is not yet implemented".format(properties.name))
        
    else:
        raise NotImplementedError("The specified tuning method '{}' is not yet implemented".format(properties.name))

    # Tune hyperparameters
    tuner.fit(X, y)
    return tuner, properties.__dict__            


    # # Random Grid Search 
    # if tune_params.name == "random_grid_search":

    #     # Set defaults if not passed 
    #     if "n_jobs" not in tune:
    #         tune_params.n_jobs = -1 # using all processors
    #     if "n_iter" not in tune:
    #         tune_params.n_iter = 5 
    #     if "cv" not in tune:
    #         tune_params.cv = 5
    #     if "refit" not in tune: 
    #         tune_params.refit = True
    #     if "scoring" not in tune: 
    #         tune_params.scoring = "f1"
        
    #     # Initialize Random Grid Search 
    #     random_grid_search = RandomizedSearchCV(estimator=model,
    #                                             scoring=tune_params.scoring,
    #                                             cv=tune_params.cv,
    #                                             refit=tune_params.refit,
    #                                             n_jobs=tune_params.n_jobs,
    #                                             n_iter=tune_params.n_iter,
    #                                             param_distributions=tune_params.param_grid)
        
    #     # Tune hyperparameters
    #     random_grid_search.fit(X, y)

    #     # Return tuned method  
    #     return random_grid_search
        

def get_tuner(method, **kwargs):
    if method == TUNE_EVOLUTIONARY_SEARCH:
        return EvolutionarySearchTuner(**kwargs)
    else:
        raise NotImplementedError("The specified tuning method '{}' is not yet implemented".format(method))
