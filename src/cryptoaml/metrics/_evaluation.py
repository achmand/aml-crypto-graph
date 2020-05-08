"""
A script which exposes all the evaluation methods used in the experiments. 
The following models are included;
"""

# TODO -> Pass labels as they may be needed in the future
# this happens when there is a case that there is only one label in both y_true and y_pred
# so just to be sure we must add labels as an argument (this is highly unlikely)

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import pandas as pd 
from collections import OrderedDict
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import pprint
from IPython.core.display import display, HTML

###### evaluation functions ###############################################
AUC              = "auc"
ACCURACY         = "accuracy"
LOG_LOSS         = "log_loss"
F1_BINARY        = "f1"
F1_MICRO         = "f1_micro"
RECALL_BINARY    = "recall"
PRECISION_BINARY = "precision"
CONFUSION_MATRIX = "confusion"

# metrics which can be displayed in a table 
TABLE_METRICS = {AUC, ACCURACY, F1_BINARY, F1_MICRO, RECALL_BINARY, PRECISION_BINARY}

def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred, normalize=True)

def compute_log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

def compute_f1_binary(y_true, y_pred):
    return f1_score(y_true, y_pred, average="binary")

def compute_f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")

def compute_recall_binary(y_true, y_pred):
    return recall_score(y_true, y_pred, average="binary")

def compute_precision_binary(y_true, y_pred):
    return precision_score(y_true, y_pred, average="binary")

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# map different metric constants to metric function
eval_options = {
    AUC:              compute_auc,
    ACCURACY:         compute_accuracy,
    F1_BINARY:        compute_f1_binary,
    F1_MICRO:         compute_f1_micro,
    RECALL_BINARY:    compute_recall_binary,
    PRECISION_BINARY: compute_precision_binary, 
    CONFUSION_MATRIX: compute_confusion_matrix
}

def evaluate(metrics, y_true, y_pred):
    
    # validate metrics argument
    m = []
    if isinstance(metrics, str):
        m.append(metrics)
    elif isinstance(metrics, list):
        m = metrics
    else:
        raise ValueError("'metrics' must be of type str or list<str>")

    # compute results for the specified metrics 
    results = OrderedDict()
    for metric in m: 
        if metric not in eval_options:
            error = "'metric'=%r is not implemented" % metric
            raise NotImplementedError(error)
        else:
            results[metric] = eval_options[metric](y_true, y_pred)

    # return results 
    return results

def results_table(results_dict, num_of_decimals=3):
    
    # create list to hold results 
    df_results = []
    
    # loop in models 
    for model, model_results in results_dict.items():
      
        # loop in feature set 
        for feature_set, set_results in model_results.items():
            
            # extract results for model per feature set 
            tmp_result = {
                "model": model + "_" + feature_set
            }
            
            # extract result for each metric
            for metric, result in set_results["metrics"].items():
                
                # check whether the current metric can be displayed in a table 
                if metric not in TABLE_METRICS:
                    continue 
                tmp_result[metric] = round(result, num_of_decimals)
            
            # add to results
            df_results.append(tmp_result)
    
    # create and return results dataframe
    df = pd.DataFrame(df_results) 
    return df 

def print_model_params(results):
    pp = pprint.PrettyPrinter(indent=4)
    for model_key, model_value in results.items():
        for feature_set, feature_set_value in model_value.items():
            print("Parameters used for '{}' on '{}' feature set".format(model_key, feature_set))
            pp.pprint(feature_set_value["params"])
            display(HTML("</hr>"))

def display_metrics_stats(results):
    for model_key, model_value in results.items():
        for feature_set, feature_set_value in model_value.items():       
            title = "'{}' on '{}' feature set - performance metrics stats".format(model_key, feature_set)
            print(title)
            display(feature_set_value["metrics_iterations"].describe())