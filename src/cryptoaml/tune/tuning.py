"""
A script which exposes all the hyperparameter tuning methods used in the experiments. 
The following methods are included;
- Random Grid Search ()
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from .. import utils as u 
from sklearn.model_selection import RandomizedSearchCV

###### Hyperparameter tuning functions ###################################
def tune_model(model, X, y, tune):

    # Validate tune arguments 
    if "param_grid" not in tune:
        raise ValueError("'param_grid' dictionary must be passed")

    tune_params = u.Namespace(tune)
    
    # Random Grid Search 
    if tune_params.name == "random_grid":

        # Set defaults if not passed 
        if "n_jobs" not in tune:
            tune_params.n_jobs = -1 # using all processors
        if "n_iter" not in tune:
            tune_params.n_iter = 5 
        if "cv" not in tune:
            tune_params.cv = 5
        if "refit" not in tune: 
            tune_params.refit = True
        if "scoring" not in tune: 
            tune_params.scoring = "f1"
        
        # Initialize Random Grid Search 
        random_grid = RandomizedSearchCV(estimator=model,
                                        cv=tune_params.cv,
                                        refit=tune_params.refit,
                                        n_jobs=tune_params.n_jobs,
                                        n_iter=tune_params.n_iter,
                                        param_distributions=tune_params.param_grid)
        
        # Tune hyperparameters
        random_grid.fit(X, y)

        # Return tuned method  
        return random_grid
        
    else:
        raise NotImplementedError("The specified tuning method '{}' is not yet implemented".format(tune))
