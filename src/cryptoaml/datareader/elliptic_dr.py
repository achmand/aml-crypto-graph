"""
Class which reads the elliptic dataset (node classification).
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

# TODO -> include verbose 
# TODO -> include validation split 

###### importing dependencies #############################################
import pandas as pd 
from .. import utils as u 

###### Elliptic data set class ############################################
class Elliptic_Dataset:
    def __init__(self, data_args, encode_classes=False):

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
                                    left_on="txId", 
                                    right_on="txId", 
                                    how="left")

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
            node_labels["class"] = node_labels["class"].apply(lambda x: -1 if x == "unknown" else 1 if x == "1" else 0)

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
        self._meta_cols = ["txId" , "ts"]
        
        # the first 94 features represent local information about the tx 
        # we skip the first one hence range(93), since that column indicates time step 
        self._lf_cols = [f"LF_{i}" for i in range(93)]

        # the ramaining 72 features represent aggregated features
        self._agg_cols = [f'AGG_{i}' for i in range(72)]

        # rename dataframe's columns 
        node_feats.columns = self._meta_cols + self._lf_cols + self._agg_cols

        # returns node features 
        return node_feats

    def get_data_split(self, train_perc, input_feats, inc_unknown=False):
        
        # first we check if we include data points with unknown label 
        if inc_unknown == False:
            unknown_label = -1 if self.encode_classes else "unknown"
            data = self.node_feats[(self.node_feats["class"] != unknown_label)].copy()
        else: 
            data = self.node_feats.copy()

        # now we make sure that the dataset is ordered by ts column 
        data.sort_values(by=["ts"], ascending=True, inplace=True)
        
        # we need to find out the timestamp we are splitting with given train_perc
        last_ts = data.tail(1)["ts"]
        split_ts = int(last_ts * train_perc)

        # split training and test by timestamp
        data_train = data[data["ts"] <= split_ts]
        data_test = data[data["ts"] > split_ts]

        # filter by input feats and create input splits   
        if input_feats == "LF": # local features (LF)
            data_train_X = data_train[self._lf_cols]
            data_test_X = data_test[self._lf_cols]
        elif input_feats == "AF": # local + aggregated features (AF)
            data_train_X = data_train[self._lf_cols + self._agg_cols]
            data_test_X = data_test[self._lf_cols + self._agg_cols]
        else:
            raise NotImplementedError("input_feats passed not yet implemented.")

        # create output split 
        data_train_y = data_train["class"]
        data_test_y = data_test["class"]

        # create dictionary to hold dataset 
        datasplit = {}
        datasplit["train_X"] = data_train_X
        datasplit["train_y"] = data_train_y
        datasplit["test_X"] = data_test_X
        datasplit["test_y"] = data_test_y

        # make as object
        datasplit = u.Namespace(datasplit) 

        # return data splits
        return datasplit