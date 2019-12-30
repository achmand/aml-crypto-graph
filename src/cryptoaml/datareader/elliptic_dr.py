"""
Class which reads the elliptic dataset (node classification)
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import pandas as pd 

###### Elliptic data set class ############################################
class Elliptic_Dataset:
    def __init__(self, data_args, encode_classes=False, cat_class_feats=False):

        # 1. load node labels as df
        labels_path = "{}{}".format(data_args.folder, data_args.classes_file) 
        self.node_labels = self._load_node_labels(labels_path, encode_classes)

        # 2. load node edges as df
        edges_path = "{}{}".format(data_args.folder, data_args.edges_file) 
        self.node_edges = self._load_node_edges(edges_path)

        # 3. load node features as df
        feats_path = "{}{}".format(data_args.folder, data_args.feats_file)
        self.node_feats = self._load_node_feats(feats_path)

        # 4. if cat_class_feats is set to true concatenate labels with features dataframe 
        if cat_class_feats:
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
        self._agg_cols = [f'agg_feat_{i}' for i in range(72)]

        # rename dataframe's columns 
        node_feats.columns = self._meta_cols + self._lf_cols + self._agg_cols

        # returns node features 
        return node_feats
