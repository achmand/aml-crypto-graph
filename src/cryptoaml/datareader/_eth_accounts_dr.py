"""
Class which reads the ethereum illicit/licit dataset (account classification).
References:  https://github.com/sfarrugia15/Ethereum_Fraud_Detection
             https://www.sciencedirect.com/science/article/abs/pii/S0957417420301433#ec-research-data
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import pandas as pd
from .. import utils as u 
from collections import OrderedDict
from ._datareader import _BaseDatareader
from sklearn.model_selection import train_test_split

###### Ethereum Accounts Classification dataset ###########################
class Eth_Accounts_Dataset(_BaseDatareader):

    # constructor ---------------------------------------------------------
    def __init__(self, 
                 data_args):
        # call parent constructor 
        super().__init__(data_args=data_args)
    
    # properties ----------------------------------------------------------
    @property
    def labels_(self):
        """Gets the label and it's respective encoding."""
        return {self.LABEL_LICIT:  0, 
                self.LABEL_ILLICIT: 1}

    # load data functions -------------------------------------------------
    def _load_dataset(self, data_args):

        # 1. load dataset 
        data_path = "{}{}".format(data_args.folder, data_args.accounts_file)
        dataset = pd.read_csv(data_path, index_col="Index") 
        dataset.index.name = None

        # 2. drop features which were not utilised in the original study,
        # keep the ones with the exact names as listed in the paper 
        dataset.drop([
            "ERC20_uniq_sent_addr.1",       # in the dataset provided this feature was listed twice 
            "Address",
            "ERC20_avg_time_between_rec_2_tnx",
            "ERC20_min_val_sent_contract",
            "ERC20_max_val_sent_contract",
            "ERC20_avg_val_sent_contract"], axis=1, inplace=True)
        
        # 3. rename 'FLAG' column to COL_CLASS constant
        dataset.rename(columns = {"FLAG": self.COL_CLASS}, inplace = True) 
        
        # 4. change column names to lower case 
        dataset.columns = map(str.lower, dataset.columns)

        # 5. change nan values to 0, and UNKNOWN in categorical fields 
        dataset.replace(r'^\s*$', np.nan, regex=True, inplace=True) # replace field that's entirely space (or empty) with NaN
        dataset[["erc20_most_sent_token_type", "erc20_most_rec_token_type"]] = dataset[[
            "erc20_most_sent_token_type", "erc20_most_rec_token_type"]].fillna(value="UNKNOWN")
        dataset.fillna(0, inplace=True)

        # 6. handle categorical features: "erc20_most_sent_token_type", "erc20_most_rec_token_type",
        # we will utilise label encoding
        dataset["erc20_most_sent_token_type"] = dataset["erc20_most_sent_token_type"].astype("category").cat.codes
        dataset["erc20_most_rec_token_type"] = dataset["erc20_most_rec_token_type"].astype("category").cat.codes

        # 7. re-arrange order of columns 
        columns = dataset.columns
        self._cols_features = [col for col in columns if col not in [self.COL_CLASS]]
        reordered_cols =  self._cols_features + [self.COL_CLASS]
        dataset = dataset[reordered_cols]

        return dataset

    def train_test_split(self, train_size, as_dict=True, shuffle=True, random_state=42):

        # split dataset, in a stratified fashion based on the class labels 
        X = self._dataset[self._cols_features]
        y = self._dataset["class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            train_size=train_size, 
                                                            random_state=random_state,
                                                            stratify=y, 
                                                            shuffle=shuffle)
        
        
        # create dictionary to hold dataset 
        datasplit = {}
        datasplit["train_X"] = X_train
        datasplit["train_y"] = y_train
        datasplit["test_X"] = X_test
        datasplit["test_y"] = y_test

        # make as object
        datasplit = u.Namespace(datasplit) 

        # check if it should be returned as dictionary or dataplit
        if as_dict == True:
            data = OrderedDict()
            data["ALL"] = datasplit
            return data
        else:
            return datasplit