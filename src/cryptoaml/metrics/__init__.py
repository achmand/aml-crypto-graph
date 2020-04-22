"""
Top level API for metrics and plotting.
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from ._evaluation import evaluate
from ._evaluation import results_table
from ._evaluation import print_model_params
from ._evaluation import display_metrics_stats

from ._plotter import plot_metric_dist
from ._plotter import plot_feature_imp
from ._plotter import plot_result_matrices
from ._plotter import plot_confusion_matrix
from ._plotter import plot_time_indexed_results
from ._plotter import elliptic_time_indexed_results

###### exposed functionality ##############################################
__all__ = ["evaluate", 
          "results_table",   
          "plot_metric_dist",
          "plot_feature_imp",
          "print_model_params",
          "plot_result_matrices",
          "plot_confusion_matrix", 
          "elliptic_time_indexed_results",
          "plot_time_indexed_results"]
###########################################################################
