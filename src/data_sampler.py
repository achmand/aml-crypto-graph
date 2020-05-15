# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import sys 
import yaml
import uuid
import argparse
import numpy as np
import pandas as pd
from logger import Logger
from collections import Counter
from cryptoaml.utils import Namespace

# datasets
import cryptoaml.datareader as cdr
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import NeighbourhoodCleaningRule

def undersample(args):
    ncr = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
    if args.data == "elliptic":
        elliptic_args = Namespace(args.elliptic_args)

        elliptic_data = cdr.get_data(source="elliptic", 
                                    config_file=args.data_config_file,
                                    encode_classes=elliptic_args.encode_classes)

        dataset = elliptic_data.train_test_split(train_size=elliptic_args.train_size,
                                                 feat_set="AF_NE",
                                                 inc_meta=True,
                                                 inc_unknown=elliptic_args.inc_unknown)

        train_X = dataset.train_X
        train_y = dataset.train_y
        test_X = dataset.test_X
        
        counter = Counter(train_y)
        print("Train set counter [Label]: {}".format(counter))
        
        if args.stratify_timestep == False:
            _, y = ncr.fit_resample(train_X[elliptic_data.feature_cols_AF_NE_], train_y)
            counter = Counter(y)
            print("Train set counter after NCL [Label]: {}".format(counter))

            indices = ncr.sample_indices_
            samples_kept = train_X.iloc[indices]
            undersampled_set = samples_kept.append(test_X, ignore_index=True)
            undersampled_set.drop(elliptic_data.feature_cols_NE_, inplace=True, axis=1)
            undersampled_set.to_csv(args.output_file, index=False, header=False)
        # stratify on time version
        else:
            tmp_data = train_X.copy()
            tmp_data["label"] = train_y.copy()
            ts_data = tmp_data.groupby("ts")

            removed = 0
            total_pre = tmp_data.shape[0]
            undersampled_set = pd.DataFrame() 
            for ts, group in ts_data:   
                
                grouped_X = group.iloc[:,:-1]
                ts_X = grouped_X[elliptic_data.feature_cols_AF_NE_]
                ts_y = group["label"]   
                counter = Counter(ts_y)
                print("Train set (ts:{}) counter Label: {}".format(ts, counter))

                X, y = ncr.fit_resample(ts_X, ts_y)  
                indices = ncr.sample_indices_
            
                counter = Counter(y)
                print("Train set (ts:{}) counter after NCR Label: {}".format(ts, counter))
                
                total_removed = ts_X.shape[0] - X.shape[0]
                print("Total removed (ts:{}): {}".format(ts, total_removed))
                removed += total_removed 
                
                samples_kept = grouped_X.iloc[indices]
                print("Total samples kept (ts:{}): {}".format(ts, samples_kept.shape[0]))
                
                undersampled_set = undersampled_set.append(samples_kept, ignore_index=True)
                
            print("-------------------------------------")
            print("Total samples removed: {} from {}".format(removed, total_pre))
            undersampled_set = undersampled_set.append(test_X, ignore_index=True)
            undersampled_set.drop(elliptic_data.feature_cols_NE_, inplace=True, axis=1)
            undersampled_set.to_csv(args.output_file, index=False, header=False)

    else:
        raise NotImplementedError("'{}' dataset not yet implemented".format(args.data))

def oversample(args):
    sm = SMOTE()
    if args.data == "elliptic":
        load_edges = args.elliptic_args.get("load_edges", True)
        elliptic_args = Namespace(args.elliptic_args)

        elliptic_data = cdr.get_data(source="elliptic", 
                                    config_file=args.data_config_file,
                                    load_edges=load_edges, 
                                    encode_classes=elliptic_args.encode_classes)

        dataset = elliptic_data.train_test_split(train_size=elliptic_args.train_size,
                                                 feat_set="AF_NE",
                                                 inc_meta=True,
                                                 inc_unknown=elliptic_args.inc_unknown)

        train_X = dataset.train_X
        train_y = dataset.train_y
        test_X = dataset.test_X
        test_y = dataset.test_y

        counter = Counter(train_y)
        print("Train set counter [Label]: {}".format(counter))
        X, y =  sm.fit_resample(train_X[elliptic_data.feature_cols_AF_NE_], train_y)
        counter = Counter(y)
        print("Train set counter after SMOTE [Label]: {}".format(counter))
        
        X["class"] = y
        X.sort_values(by=["ts"], inplace=True)
        X.insert(0, "txId", np.arange(len(X)))
        test_X["class"] = test_y
        oversampled_set = X.append(test_X, ignore_index=True)
        oversampled_set["txId"] = [uuid.uuid4() for _ in range(len(oversampled_set))]
        oversampled_set["txId"] =  oversampled_set["txId"].astype(str)
        
        tx_classes = oversampled_set[["txId", "class"]]
        tx_embeddings = oversampled_set[["txId"] + elliptic_data.feature_cols_NE_]
        tx_features = oversampled_set[["txId"] + elliptic_data.feature_cols_AF_]

        tx_features.to_csv(args.output_dir + "elliptic_txs_features.csv", index=False, header=False)
        tx_classes.to_csv(args.output_dir + "elliptic_txs_classes.csv", index=False, header=True)
        tx_embeddings.to_csv(args.output_dir + "elliptic_embs.csv", index=False, header=True)

###### start under-sampling the dataset using NCR #########################
if __name__ == "__main__":

    # create parser 
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config_file", 
                        type=argparse.FileType(mode="r"),
                        help="optional, yaml file containing params for experiment result extraction")

    # parse arguments from config file 
    args = parser.parse_args()
    if args.config_file:
        properties = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in properties.items():
            arg_dict[key] = value

    # create logger 
    logger_exp = Logger(args.save_log, args.log_path)
    logger_exp.info("-------- START SAMPLING --------")
    logger_exp.info("- PROPERTIES -")
    logger_exp.info (args.__dict__, pp=True) 
    
    try:
        
        # undersample dataset using NCL 
        if args.sample_type == "ncl":
            undersample(args)
        # oversample dataset using SMOTE
        elif args.sample_type == "smote":
            oversample(args) 
        else:
            raise NotImplementedError("'{}' sampling technique not yet implemented".format(args.sample_type))

        logger_exp.info("-------- END SAMPLING --------")
    except:
        e = sys.exc_info()
        logger_exp.exception(e)     


