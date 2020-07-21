# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import sys 
import yaml
import argparse
import pandas as pd
from logger import Logger
from collections import OrderedDict
from cryptoaml.utils import Namespace, save_pickle

# datasets
import cryptoaml.datareader as cdr

# models 
from cryptoaml.models import XgboostAlgo
from cryptoaml.models import LightGbmAlgo
from cryptoaml.models import CatBoostAlgo
from cryptoaml.models import RandomForestAlgo
from catboost import CatBoostClassifier

###### load dataset #######################################################
def build_dataset(args):

    logger_exp.info("--- START BUILDING DATASET ---")
    logger_exp.info("- DATASET -")
    logger_exp.info(args.data)
    logger_exp.info("- CONFIG -")
    logger_exp.info(args.data_config_file, pp=True)
    
    dataset=None
    if args.data == "elliptic":
        load_edges = args.elliptic_args.get("load_edges", True)
        elliptic_args = Namespace(args.elliptic_args)
        elliptic_data = cdr.get_data(source="elliptic", 
                                     config_file=args.data_config_file,
                                     load_edges=load_edges, 
                                     encode_classes=elliptic_args.encode_classes)
                                     
        dataset = elliptic_data.train_test_split(train_size=elliptic_args.train_size,
                                                 feat_set=elliptic_args.feat_sets,
                                                 inc_meta=False,
                                                 inc_unknown=elliptic_args.inc_unknown)
    elif args.data == "eth_accounts":
        eth_accounts_args = Namespace(args.eth_accounts_args)
        eth_accounts_data = cdr.get_data(source="eth_accounts", 
                                         config_file=args.data_config_file)
        dataset = eth_accounts_data.train_test_split(train_size=eth_accounts_args.train_size)     

    elif args.data == "noaa_weather":
        noaa_args = Namespace(args.noaa_args)           
        noaa_data = cdr.get_data(source="noaa_weather", 
                                         config_file=args.data_config_file)         
        dataset = noaa_data.train_test_split(train_size=noaa_args.train_size)     

    else:
        raise NotImplementedError("'{}' dataset not yet implemented".format(args.data))
    
    logger_exp.info("--- FINISH BUILDING DATASET ---")
    return dataset

###### build models #######################################################
def load_model(model_type, path=None, params=None):
    model = None 
    model_args = {} 
    if path != None: 
        model_args["persist_props"] = {"method":"load", "load_path": path}
    
    if params != None:
        model_args = {**model_args, **params}

    if model_type == "random_forest":
        model = RandomForestAlgo(**model_args)
    elif model_type == "xg_boost":
        model = XgboostAlgo(**model_args)
    elif model_type == "light_boost":
        model = LightGbmAlgo(**model_args)
    elif model_type == "cat_boost":
        model = CatBoostAlgo(**model_args)
    else:
        raise NotImplementedError("'{}' model not yet implemented".format(model))

    return model

def build_models(args):
    logger_exp.info("--- START BUILDING MODELS ---")
    logger_exp.info("- MODELS -")
    logger_exp.info(args.models)
    models = OrderedDict()

    for model in args.models:
        logger_exp.info("[START BUILDING MODEL '{}']".format(model))
        logger_exp.info("- ARGUMENTS -")
        model_args = args.models[model]
        logger_exp.info(model_args, pp=True)    

        # get model type 
        model_type = model_args["model"]
        if model_type not in models:
            models[model_type] = OrderedDict()

        # get model feature set 
        model_feat_set = model_args["feat_set"]
        if model_feat_set not in models: 
            model_path = model_args.get("load_path", None)
            params = model_args.get("params", None)
            models[model_type][model_feat_set] = {}
            models[model_type][model_feat_set]["model"] = load_model(model_type, model_path, params)
            models[model_type][model_feat_set]["iterations"] = model_args["iterations"]

        logger_exp.info("[FINISH BUILDING MODEL '{}']".format(model))
    
    logger_exp.info("--- FINISH BUILDING MODELS ---")
    return models 

###### extract results #####################################################
def extract_time_indexed(metric, dataset, model, X, y):
    results = []
    if dataset == "elliptic" or dataset == "noaa_weather":
        tmp_data = X.copy()
        tmp_data["label"] = y.copy()
        ts_data = tmp_data.groupby("ts")
        for ts, group in ts_data:
            test_ts_X = group.iloc[:,:-1]
            test_ts_y = group["label"]
            evaluation = model.evaluate([metric], test_ts_X, test_ts_y)
            label_count = group["label"].value_counts()
            results.append({"timestep": ts, "score":evaluation[metric], "total_pos_label": label_count.tolist()[1]}) 
        return results
    else:
        raise NotImplementedError("'{}' dataset cannot extract time indexed score".format(model))

def extract_results(args, models, dataset):
    logger_exp.info("--- START GATHERING RESULTS ---")
    evaluation_metrics = args.evaluation_metrics

     # check if time indexed evaluation is required 
    time_indexed_metric = args.time_indexed_evaluation
    
    # loop models 
    results = OrderedDict()
    for model_key in models: 
        results[model_key] = OrderedDict()
        for feat_set in models[model_key]:
            results[model_key][feat_set] = OrderedDict()
            logger_exp.info("----> [START GATHERING FOR '{}' MODEL FOR '{}' FEATURE SET]".format(model_key, feat_set))
            
            # get train and test set 
            train_X = dataset[feat_set].train_X
            train_y = dataset[feat_set].train_y
            test_X  = dataset[feat_set].test_X
            test_y  = dataset[feat_set].test_y

            # get model and iterations 
            model = models[model_key][feat_set]["model"]
            iterations = models[model_key][feat_set]["iterations"]
          
            # loop number of iterations specified, 
            # this is mainly set greater than 1 when 
            # evaluating a non-deterministic model (results are averaged)
            results[model_key][feat_set]["metrics_iterations"] = []
            results[model_key][feat_set]["feat_importance_iterations"] = []
            results[model_key][feat_set]["time_indexed_iterations"] = []
            for i in range(iterations):
                logger_exp.info("- CURRENT ITERATION ({}) -".format(i))
                if model_key == "light_boost":
                    model.set_params(**{"random_state": i, "n_jobs": 16})
                    print(model.get_params())
                elif model_key == "xg_boost":
                    model.set_params(**{"random_state": i, "n_jobs": 16})
                    print(model.get_params())
                elif model_key == "cat_boost":
                    tmp_cat = CatBoostClassifier(**model._model._init_params)
                    tmp_cat.set_params(random_seed=i)
                    model._model = tmp_cat
                   
                model.fit(train_X, train_y, tune=False)

                # first iteration extract parameters 
                if i == 0:
                    results[model_key][feat_set]["params"] = model.get_params()

                # evaluate model and extract metrics 
                current_metrics = {}
                current_metrics["iteration"] = i               
                # print(model.evaluate(evaluation_metrics, test_X, test_y))
                current_metrics = {**current_metrics, **model.evaluate(evaluation_metrics, test_X, test_y)}
                results[model_key][feat_set]["metrics_iterations"].append(current_metrics)

                # extract feature importance 
                results[model_key][feat_set]["feat_importance_iterations"].append(model.feature_importances_)

                # extracted time indexed evaluation if required
                if time_indexed_metric != "none":
                    current_time_indexed = extract_time_indexed(time_indexed_metric, args.data, model, test_X, test_y)
                    results[model_key][feat_set]["time_indexed_iterations"].append(current_time_indexed)

            # summarise metrics 
            if iterations == 1: # no need to average results 
                metric_single = results[model_key][feat_set]["metrics_iterations"][0]
                results[model_key][feat_set]["metrics"] = { key: metric_single[key] for key in evaluation_metrics }
                results[model_key][feat_set]["importance"] = results[model_key][feat_set]["feat_importance_iterations"][0]
                if time_indexed_metric != "none":
                    results[model_key][feat_set]["time_metrics"] = pd.DataFrame(results[model_key][feat_set]["time_indexed_iterations"][0])
         
            else: # average results since more than 1 iteration
                metrics_all = results[model_key][feat_set]["metrics_iterations"]
                results[model_key][feat_set]["metrics_iterations"] = pd.DataFrame(metrics_all)
                results[model_key][feat_set]["metrics"] = results[model_key][feat_set]["metrics_iterations"][evaluation_metrics].mean(axis=0).to_dict()
                feat_imp_all = results[model_key][feat_set]["feat_importance_iterations"]
                results[model_key][feat_set]["feat_importance_iterations"] = pd.concat(feat_imp_all, axis=1)
                results[model_key][feat_set]["importance"] = results[model_key][feat_set]["feat_importance_iterations"].mean(
                    axis=1).to_frame().rename(columns={0:"importance"})
                if time_indexed_metric != "none":
                    time_indexed_dicts = results[model_key][feat_set]["time_indexed_iterations"]
                    time_indexed_dfs = []
                    for time_indexed_i in time_indexed_dicts:
                        time_indexed_dfs.append(pd.DataFrame(time_indexed_i))
                    results[model_key][feat_set]["time_indexed_iterations"] = time_indexed_dfs                 
                    time_indexed_merged = pd.concat(time_indexed_dfs, axis=1)
                    tmp_df = time_indexed_merged["score"].mean(axis=1).to_frame().rename(columns={0:"score"})
                    tmp_df["timestep"] = time_indexed_dfs[0]["timestep"]
                    tmp_df["total_pos_label"] = time_indexed_dfs[0]["total_pos_label"]
                    results[model_key][feat_set]["time_metrics"] = tmp_df

            logger_exp.info("----> [FINISH GATHERING FOR '{}' MODEL FOR '{}' FEATURE SET]".format(model_key, feat_set))

    logger_exp.info("--- END GATHERING RESULTS ---")
    return results

###### start result extraction ############################################
if __name__ == "__main__":

    # create parser 
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config_file", 
                        default="configuration/experiments/1_results_extraction.yaml", 
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
    logger_exp.info("-------- START RESULTS EXTRACTION --------")
    logger_exp.info("- PROPERTIES -")
    logger_exp.info (args.__dict__, pp=True) 
    try:
        
        # build dataset 
        dataset = build_dataset(args)

        # build models 
        models = build_models(args)       

        # extract results 
        results = extract_results(args, models, dataset)
        
        # save results 
        results_path = args.results_path
        file_name = args.results_file
        save_pickle("{}/{}".format(results_path, file_name), results)

        logger_exp.info("-------- FINISH RESULTS EXTRACTION --------")
    except:
        e = sys.exc_info()
        logger_exp.exception(e)     


