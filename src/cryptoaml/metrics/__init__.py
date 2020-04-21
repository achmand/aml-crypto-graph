"""
Top level API for metrics and plotting.
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from ._evaluation import evaluate
from ._evaluation import results_table
from ._plotter import plot_confusion_matrix
from ._plotter import plot_time_indexed_results

###### exposed functionality ##############################################
__all__ = ["evaluate", "results_table", "plot_confusion_matrix", "plot_time_indexed_results"]
###########################################################################
