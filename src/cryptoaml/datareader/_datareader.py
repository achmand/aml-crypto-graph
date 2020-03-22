"""
Base class for datasets. 
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from abc import ABC, abstractmethod

###### base datareader ####################################################
class _BaseDatareader(ABC):
    """
    Base class for datasets.
    Warning: This class should not be used directly. Use derived classes instead.
    """

    # constants -----------------------------------------------------------
    # columns - meta
    COL_CLASS = "class"  

    # labels 
    LABEL_LICIT   = "licit"   
    LABEL_ILLICIT = "illicit"

    # constructor ---------------------------------------------------------
    def __init__(self, data_args, **kwargs):
        
        # set defaults 
        self._cols_features = []
        
        # loads dataset from source
        self._dataset = self._load_dataset(data_args, **kwargs)
            
    # properties ----------------------------------------------------------
    @property
    def dataset_(self):
        """Get the current dataset."""
        return self._dataset      

    @property
    def feature_cols_(self):
        """Get the feature names from the current dataset."""
        return self._cols_features      

    @property
    @abstractmethod 
    def labels_(self):
        """Gets the label and it's respective encoding."""
        pass 

    @property 
    def label_count_(self):
        """Gets the labels and their respective count."""
        return self._dataset[self.COL_CLASS].value_counts()
        
    # load data functions -------------------------------------------------
    @abstractmethod 
    def _load_dataset(self, data_args, **kwargs):
        pass
