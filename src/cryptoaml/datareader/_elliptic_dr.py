"""
Class which reads the elliptic dataset (node/transaction classification).
References: https://www.elliptic.co/
            https://arxiv.org/abs/1908.02591
            https://www.kaggle.com/ellipticco/elliptic-data-set
            https://www.elliptic.co/our-thinking/elliptic-dataset-cryptocurrency-financial-crime
            https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14
            https://mitibmwatsonailab.mit.edu/research/projects/scalable-graph-learning-for-anti-money-laundering/
            https://www.markrweber.com/graph-deep-learning
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import pandas as pd 
from .. import utils as u 
from collections import OrderedDict
from ._datareader import _BaseDatareader

###### elliptic dataset ###################################################
class Elliptic_Dataset(_BaseDatareader):
    
    # constants -----------------------------------------------------------
    # columns - meta 
    COL_TXID  = "txId"  # TX ID 
    COL_TS    = "ts"    # timestep 

    # columns - feature sets 
    FEATS_LF    = "LF"       # Local features
    FEATS_AF    = "AF"       # Local features + Aggregated Features
    FEATS_LF_NE = "LF_NE"    # Local features and Node Embeddings from GCN 
    FEATS_AF_NE = "AF_NE"    # Local features + Aggregated Features and Node Embedding from GCN 

    # labels
    LABEL_UNKNOWN = "unknown" # Unknown label 

    # constructor ---------------------------------------------------------
    def __init__(self, 
                 data_args, 
                 encode_classes=True,
                 load_edges=True):
        
        # call parent constructor 
        super().__init__(data_args=data_args, 
                         encode_classes=encode_classes,
                         load_edges=load_edges)

    # properties ----------------------------------------------------------
    @property
    def edges_(self):
        return self._node_edges

    @property 
    def node_embs_(self):
        return self._node_embs

    @property
    def labels_(self):
        """Gets the label and it's respective encoding."""
        return {self.LABEL_LICIT:   self._label_licit, 
                self.LABEL_ILLICIT: self._label_illicit,
                self.LABEL_UNKNOWN: self._label_unknown}

    @property
    def feature_cols_LF_(self):
        """Get the feature names for the Local feature set (LF)."""
        return self._cols_lf      

    @property
    def feature_cols_AF_(self):
        """Get the feature names for the All feature set (AF)."""
        return self.feature_cols_      

    @property
    def feature_cols_NE_(self):
        """Get the feature names for the node embeddings feature set (NE)."""
        return self._cols_ne      

    @property
    def feature_cols_LF_NE_(self):
        """Get the feature names for the Local feature set + node embeddings (LF_NE)."""
        return self._cols_lf + self._cols_ne     

    @property
    def feature_cols_AF_NE_(self):
        """Get the feature names for the All feature set + node embeddings (AF_NE)."""
        return  self.feature_cols_ + self._cols_ne     

    def get_feature_names(self, feat_set, inc_meta=False):

        feat_set_cols = self._cols_meta.copy() if inc_meta else []
        if feat_set == self.FEATS_LF:
            feat_set_cols.extend(self.feature_cols_LF_)
        elif feat_set == self.FEATS_LF_NE:
            feat_set_cols.extend(self.feature_cols_LF_NE_)
        elif feat_set == self.FEATS_AF:
            feat_set_cols.extend(self.feature_cols_AF_)
        elif feat_set == self.FEATS_AF_NE:
            feat_set_cols.extend(self.feature_cols_AF_NE_)

        return feat_set_cols

    # load data functions -------------------------------------------------
    def _load_dataset(self, data_args, encode_classes, load_edges=True):

        # 1. load node labels as dataframe
        labels_path = "{}{}".format(data_args.folder, data_args.classes_file) 
        node_labels = self._load_node_labels(labels_path, encode_classes)

        # 2. load node edges as dataframe
        if load_edges == True:
            edges_path = "{}{}".format(data_args.folder, data_args.edges_file) 
            self._node_edges = self._load_node_edges(edges_path)

        # 3. load node features as dataframe
        feats_path = "{}{}".format(data_args.folder, data_args.feats_file)
        node_feats = self._load_node_feats(feats_path)

        # 4. load node embeddings extracted from GCN as dataframe 
        embs_paths = "{}{}".format(data_args.folder, data_args.embs_file)
        self._node_embs = self._load_node_embs(embs_paths)

        # 5. concatenate labels with features dataframe
        dataset = pd.merge(node_feats, 
                           self._node_embs, 
                           left_on=self.COL_TXID, 
                           right_on=self.COL_TXID, 
                           how="left")

        # 6. concatenate labels with features dataframe     
        dataset = pd.merge(dataset, 
                           node_labels, 
                           left_on=self.COL_TXID, 
                           right_on=self.COL_TXID, 
                           how="left")

        return dataset

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
        self._cols_meta = [self.COL_TXID]
        
        # the first 94 features represent local information about the tx 
        # we skip the first one hence range(93), since that column indicates time step 
        self._cols_lf = [self.COL_TS] + [f"LF_{i}" for i in range(93)]

        # the ramaining 72 features represent aggregated features
        self._cols_agg = [f"AGG_{i}" for i in range(72)]

        # rename dataframe's columns 
        node_feats.columns = self._cols_meta + self._cols_lf + self._cols_agg

        # take reference of feature set columns 
        self._cols_features = self._cols_lf + self._cols_agg

        # returns node features 
        return node_feats

    def _load_node_embs(self, path):

        # reads from csv file 
        node_embs = pd.read_csv(path)

        # take reference for node embedding columns 
        self._cols_ne = node_embs.columns[1:].values.tolist()
        
        # returns the node embeddings extracted from GCN 
        return node_embs 

    def train_test_split(self, train_size, feat_set, inc_meta=False, inc_unknown=False):
        
        # feat set check if list 
        feat_set_list_type = type(feat_set) is list
        
        # first we check if we include data points with unknown label 
        if inc_unknown == False:
            data = self._dataset[(self._dataset[self.COL_CLASS] != self._label_unknown)].copy()
        else: 
            data = self._dataset.copy()

        # now we make sure that the dataset is ordered by ts column 
        data.sort_values(by=[self.COL_TS], ascending=True, inplace=True)
        
        # we need to find out the timestamp we are splitting with given train_size
        last_ts = data.tail(1)[self.COL_TS]
        split_ts = int(last_ts * train_size)

        # split training and test by timestamp (temporal splitting)
        data_train = data[data[self.COL_TS] <= split_ts]
        data_test = data[data[self.COL_TS] > split_ts]

        # ordered dict as a list of str was passed as 'feat_set'
        if feat_set_list_type == True:
            data = OrderedDict()
            for features in feat_set:
                data[features] = self._get_feat_set(features,
                                                    data_train,
                                                    data_test,
                                                    inc_meta)
            return data 
            
        # one data split as a str was passed as 'feat_set'
        datasplit = self._get_feat_set(feat_set, 
                                       data_train, 
                                       data_test,
                                       inc_meta)
        return datasplit                

    def _get_feat_set(self, feat_set, data_train, data_test, inc_meta):

        # filter by input feats and create input splits   
        feat_set_cols = self._cols_meta.copy() if inc_meta else []
        
        # Local feature set (LF)
        if feat_set == self.FEATS_LF: 
            feat_set_cols.extend(self._cols_lf)
        # All feature set (AF)
        elif feat_set == self.FEATS_AF: 
            feat_set_cols.extend(self._cols_lf + self._cols_agg)
        # Local feature set + node embeddings (LF_NE)
        elif feat_set == self.FEATS_LF_NE:
            feat_set_cols.extend(self._cols_lf + self._cols_ne)
        # All feature set + node embeddings (AF_NE)
        elif feat_set == self.FEATS_AF_NE:
            feat_set_cols.extend(self._cols_lf + self._cols_agg + self._cols_ne)
        else:
            raise NotImplementedError("'input_feats' passed not yet implemented.")

        # create input split 
        data_train_X = data_train[feat_set_cols]
        data_test_X = data_test[feat_set_cols]

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