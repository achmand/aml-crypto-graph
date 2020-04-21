# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import sys 
import yaml
import argparse
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

###### load dataset #######################################################
def build_dataset(args):

    logger_exp.info("--- START BUILDING DATASET ---")
    logger_exp.info("- DATASET -")
    logger_exp.info(args.data)
    logger_exp.info("- CONFIG -")
    logger_exp.info(args.data_config_file, pp=True)
    
    dataset=None
    if args.data == "elliptic":
        elliptic_args = Namespace(args.elliptic_args)
        elliptic_data = cdr.get_data(source="elliptic", 
                                     config_file=args.data_config_file,
                                     encode_classes=elliptic_args.encode_classes)
        dataset = elliptic_data.train_test_split(train_size=elliptic_args.train_size,
                                                 feat_set=elliptic_args.feat_sets,
                                                 inc_meta=False,
                                                 inc_unknown=elliptic_args.inc_unknown)
    else:
        raise NotImplementedError("'{}' dataset not yet implemented".format(args.data))
    
    logger_exp.info("--- FINISH BUILDING DATASET ---")
    return dataset

###### build models #######################################################
# TODO -> HANDLE ARGS UKOLL
def load_model(model_type, path=None):
    model = None 
    model_args = {} 
    if path != None: 
        model_args["persist_props"] = {"method":"load", "load_path": path}

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
            models[model_type][model_feat_set] = {}
            models[model_type][model_feat_set]["model"] = load_model(model_type, model_path)
            models[model_type][model_feat_set]["iterations"] = model_args["iterations"]

        logger_exp.info("[FINISH BUILDING MODEL '{}']".format(model))
    
    logger_exp.info("--- FINISH BUILDING MODELS ---")
    return models 

###### extract results #####################################################
def extract_results(args, models, dataset):
    logger_exp.info("--- START GATHERING RESULTS ---")
    evaluation_metrics = args.evaluation_metrics
    results = OrderedDict()
    
    # loop models 
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
            for i in range(iterations):
                logger_exp.info("- CURRENT ITERATION ({}) -".format(i))
                
                # train model 
                model.fit(train_X, train_y, tune=False)
                              
                # first iteration extract parameters 
                if i == 0:
                    results[model_key][feat_set]["params"] = model.get_params()

                # evaluate model and extract metrics 
                current_metrics = {}
                current_metrics["iteration"] = i
                current_metrics = {**current_metrics, **model.evaluate(evaluation_metrics, test_X, test_y)}
                results[model_key][feat_set]["metrics_iterations"].append(current_metrics)

                # extract feature importance 
                results[model_key][feat_set]["feat_importance_iterations"].append(model.feature_importances_)

            # summarise metrics 
            if iterations == 1: # no need to average results 
                metric_single = results[model_key][feat_set]["metrics_iterations"][0]
                results[model_key][feat_set]["metrics"] = { key: metric_single[key] for key in evaluation_metrics }
                results[model_key][feat_set]["importance"] = results[model_key][feat_set]["feat_importance_iterations"][0]
                
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


