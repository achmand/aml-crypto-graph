"""
Class which reads the elliptic dataset (node classification).
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

# https://www.elliptic.co/
# https://arxiv.org/abs/1908.02591
# https://www.kaggle.com/ellipticco/elliptic-data-set
# https://www.elliptic.co/our-thinking/elliptic-dataset-cryptocurrency-financial-crime
# https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14
# https://mitibmwatsonailab.mit.edu/research/projects/scalable-graph-learning-for-anti-money-laundering/
# https://www.markrweber.com/graph-deep-learning

###### Importing dependencies #############################################
import pandas as pd 
from .. import utils as u 
from collections import OrderedDict

###### Elliptic data set class ############################################
class Elliptic_Dataset:
    
    # Constants -----------------------------------------------------------
    # Columns 
    COL_TXID  = "txId"  # TX ID 
    COL_TS    = "ts"    # TX time step 
    COL_CLASS = "class" # Class/Label of the TX

    # Feature sets 
    FEATS_LF = "LF" # Local features
    FEATS_AF = "AF" # Local features + Aggregated Features
   
    # Labels
    LABEL_UNKNOWN = "unknown" # Unknown label 
    LABEL_LICIT   = "licit"   # TX created by a licit node (exchanges, miners, etc...)
    LABEL_ILLICIT = "illicit" # TX created by an illicit node (scams, terrorist, etc...)

    # Constructor ---------------------------------------------------------
    def __init__(self, data_args, encode_classes=True):

        # set encode classes attribute
        self.encode_classes = encode_classes

        # 1. load node labels as df
        labels_path = "{}{}".format(data_args.folder, data_args.classes_file) 
        self.node_labels = self._load_node_labels(labels_path, encode_classes)

        # 2. load node edges as df
        edges_path = "{}{}".format(data_args.folder, data_args.edges_file) 
        self.node_edges = self._load_node_edges(edges_path)

        # 3. load node features as df
        feats_path = "{}{}".format(data_args.folder, data_args.feats_file)
        self.node_feats = self._load_node_feats(feats_path)

        # 4. concatenate labels with features dataframe     
        self.node_feats = pd.merge(self.node_feats, 
                                    self.node_labels, 
                                    left_on=self.COL_TXID, 
                                    right_on=self.COL_TXID, 
                                    how="left")

    # Properties ----------------------------------------------------------
    @property
    def labels(self):
        return {self.LABEL_LICIT:   self._label_licit, 
                self.LABEL_ILLICIT: self._label_illicit,
                self.LABEL_UNKNOWN: self._label_unknown}

    @property 
    def label_count(self):
        return self.node_labels[self.COL_CLASS].value_counts()

    def _load_node_labels(self, path, encode_classes):

        # reads from csv file 
        node_labels = pd.read_csv(path)

        # encode classes, originally the dataset uses the following encoding 
        # labelled as "1" = illicit 
        # labelled as "2" = licit 
        # labelled as "unknown" = unknown label
        
        # if encode_classes is set to true it will use the following encoding 
        # labelled as "0" = licit 
        # labelled as "1" = illicit 
        # labelled as "-1" = unknown label 
        if encode_classes:
            self._label_licit = 0
            self._label_illicit = 1
            self._label_unknown = -1
            node_labels[self.COL_CLASS] = node_labels[self.COL_CLASS].apply(
                lambda x: -1 if x == self.LABEL_UNKNOWN else 1 if x == "1" else 0)
        else:
            self._label_licit = 2
            self._label_illicit = 1
            self._label_unknown = self.LABEL_UNKNOWN

        # returns node id and labels 
        return node_labels

    def _load_node_edges(self, path):

        # reads from csv file 
        node_edges = pd.read_csv(path)
        
        # returns node id (source) and node id (target)
        return node_edges

    def _load_node_feats(self, path):
        
        # reads from csv file 
        node_feats = pd.read_csv(path, header=None)
        
        # rename column names 
        self._cols_meta = [self.COL_TXID, self.COL_TS]
        
        # the first 94 features represent local information about the tx 
        # we skip the first one hence range(93), since that column indicates time step 
        self._cols_lf = [f"LF_{i}" for i in range(93)]

        # the ramaining 72 features represent aggregated features
        self._cols_agg = [f"AGG_{i}" for i in range(72)]

        # rename dataframe's columns 
        node_feats.columns = self._cols_meta + self._cols_lf + self._cols_agg

        # returns node features 
        return node_feats

    def train_test_split(self, train_size, feat_set, inc_meta=False, inc_unknown=False):
        
        # first we check if we include data points with unknown label 
        if inc_unknown == False:
            data = self.node_feats[(self.node_feats[self.COL_CLASS] != self._label_unknown)].copy()
        else: 
            data = self.node_feats.copy()

        # now we make sure that the dataset is ordered by ts column 
        data.sort_values(by=[self.COL_TS], ascending=True, inplace=True)
        
        # we need to find out the timestamp we are splitting with given train_size
        last_ts = data.tail(1)[self.COL_TS]
        split_ts = int(last_ts * train_size)

        # split training and test by timestamp
        data_train = data[data[self.COL_TS] <= split_ts]
        data_test = data[data[self.COL_TS] > split_ts]

        # ordered dict as a list of str was passed as 'feat_set'
        if type(feat_set) is list:
            data = OrderedDict()
            for features in feat_set:
                data[features] = self._get_feat_set(features,
                                                    data_train,
                                                    data_test,
                                                    inc_meta)
            return data 
            
        # one data split as a str was passed as 'feat_set'
        else:
            datasplit = self._get_feat_set(feat_set, 
                                           data_train, 
                                           data_test,
                                           inc_meta)
            return datasplit                

    def _get_feat_set(self, feat_set, data_train, data_test, inc_meta):

        # filter by input feats and create input splits   
        feat_set_cols = self._cols_meta.copy() if inc_meta else []
        if feat_set == self.FEATS_LF: 
            feat_set_cols.extend(self._cols_lf)
            data_train_X = data_train[feat_set_cols]
            data_test_X = data_test[feat_set_cols]
        elif feat_set == self.FEATS_AF: 
            feat_set_cols.extend(self._cols_lf + self._cols_agg)
            data_train_X = data_train[feat_set_cols]
            data_test_X = data_test[feat_set_cols]
        else:
            raise NotImplementedError("'input_feats' passed not yet implemented.")

        # create output split 
        data_train_y = data_train[self.COL_CLASS]
        data_test_y = data_test[self.COL_CLASS]

        # create dictionary to hold dataset 
        datasplit = {}
        datasplit["train_X"] = data_train_X
        datasplit["train_y"] = data_train_y
        datasplit["test_X"] = data_test_X
        datasplit["test_y"] = data_test_y

        # make as object
        datasplit = u.Namespace(datasplit) 
        return datasplit