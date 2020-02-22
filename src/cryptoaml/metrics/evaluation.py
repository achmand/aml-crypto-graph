"""
A script which exposes all the evaluation methods used in the experiments. 
The following models are included;
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

###### Evaluation functions ###############################################
F1_BINARY        = "f1"
F1_MICRO         = "f1_micro"
RECALL_BINARY    = "recall"
PRECISION_BINARY = "precision"
CONFUSION_MATRIX = "confusion"

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

# Map different metric constants to metric function
eval_options = {
    F1_BINARY:        compute_f1_binary,
    F1_MICRO:         compute_f1_micro,
    RECALL_BINARY:    compute_recall_binary,
    PRECISION_BINARY: compute_precision_binary, 
    CONFUSION_MATRIX: compute_confusion_matrix
}

def evaluate(metrics, y_true, y_pred):
    
    # Validate metrics argument
    m = []
    if isinstance(metrics, str):
        m.append(metrics)
    elif isinstance(metrics, list):
        m = metrics
    else:
        raise ValueError("'metrics' must be of type str or list<str>")

    # Compute results for the specified metrics 
    results = {}
    for metric in m: 
        if metric not in eval_options:
            error = "'metric'=%r is not implemented" % metric
            raise NotImplementedError(error)
        else:
            results[metric] = eval_options[metric](y_true, y_pred)

    # Return results 
    return results