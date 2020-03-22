"""
Class which reads the elliptic dataset (account classification).
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

# TODO -> sort out bug from the other thesis 

# https://github.com/sfarrugia15/Ethereum_Fraud_Detection
# https://www.sciencedirect.com/science/article/abs/pii/S0957417420301433#ec-research-data

###### importing dependencies #############################################
import pandas as pd
from ._datareader import _BaseDatareader

###### Ethereum Fraud Classification dataset ##############################
class Eth_Fraud_Dataset(_BaseDatareader):

    # constants -----------------------------------------------------------
    # columns - meta
    COL_ADDRESS = "address"
    
    # constructor ---------------------------------------------------------
    def __init__(self, 
                 data_args, 
                 **kwargs):
        super().__init__(
            data_args=data_args,
            **kwargs
        )

    # properties ----------------------------------------------------------
    @property
    def labels_(self):
        """Gets the label and it's respective encoding for the 'eth_fraud' dataset."""
        return {self.LABEL_LICIT:   0,
                self.LABEL_ILLICIT: 1}

    # load data functions -------------------------------------------------
    def _load_dataset(self, data_args):

        # path to file/s 
        accounts_path = "{}{}".format(data_args.folder, data_args.accounts_file)

        # reads from csv file         
        dataset = pd.read_csv(accounts_path)
        
        # change 'FLAG' column to 'class' for consistency between datareaders
        dataset.rename(columns={"FLAG": self.COL_CLASS}, inplace=True)

        # remove the 'Index' column
        del dataset["Index"]
        
        # make column names lower case 
        dataset.columns = map(str.lower, dataset.columns)

        # take reference of feature set columns 
        columns = dataset.columns
        self._cols_features = [col for col in columns if col not in [self.COL_ADDRESS, self.COL_CLASS]]

        # re-arrange the order of the dataframe 
        reordered_cols = [self.COL_ADDRESS] + self._cols_features + [self.COL_CLASS]
        dataset = dataset[reordered_cols]
        
        return dataset
