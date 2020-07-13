# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import sys 
import yaml
import argparse
from logger import Logger
from collections import OrderedDict
from cryptoaml.utils import Namespace

# datasets
import cryptoaml.datareader as cdr

# models 
from cryptoaml.models import XgboostAlgo
from cryptoaml.models import LightGbmAlgo
from cryptoaml.models import CatBoostAlgo
from cryptoaml.models import RandomForestAlgo

# -> TODO -> does not work with load models as training is done anyways 

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
def build_models(args, dataset):
    logger_exp.info("--- START BUILDING MODELS ---")
    logger_exp.info("- MODELS -")
    logger_exp.info(args.models)

    models = OrderedDict()
    for model in args.models:

        logger_exp.info("[START BUILDING MODEL '{}']".format(model))
        logger_exp.info("- ARGUMENTS -")
        model_args = getattr(args, model + "_args")
        logger_exp.info(model_args, pp=True)    
        model_args = Namespace(model_args)

        # setup model arguments 
        arguments = {}
        if hasattr(model_args, "arguments"):
            arguments = model_args.arguments

        # setup tune properties     
        tune_props = None 
        if hasattr(model_args, "tune_props"):
            tune_props = model_args.tune_props

        # setup persistence properties 
        persist_props = None 
        persist_initial_path = ""
        persist_method_key = ""
        
        persist_model = hasattr(model_args, "persist_props") 
        if persist_model:
            persist_props = model_args.persist_props
            persist_method = persist_props["method"]
            persist_method_key = persist_method + "_path"
            persist_initial_path = persist_props[persist_method_key]

        models[model] = OrderedDict()
        for feature_set in dataset.keys():
        
            logger_exp.info("----> [START BUILDING '{}' MODEL FOR '{}' FEATURE SET]".format(model, feature_set))

            # setup persistence path according to feature set 
            if persist_model:
                persist_props[persist_method_key] = persist_initial_path + "/" + feature_set
           
            if model == "random_forest":
               models[model][feature_set] = RandomForestAlgo(tune_props=tune_props,
                                                             persist_props=persist_props, 
                                                             **arguments)    
            elif model == "xg_boost":
                models[model][feature_set] = XgboostAlgo(tune_props=tune_props,
                                                         persist_props=persist_props, 
                                                         **arguments)             
            elif model == "light_boost":
                models[model][feature_set] = LightGbmAlgo(tune_props=tune_props,
                                                          persist_props=persist_props, 
                                                          **arguments)
            elif model == "cat_boost":
                models[model][feature_set] = CatBoostAlgo(tune_props=tune_props,
                                                          persist_props=persist_props, 
                                                          **arguments) 
            else:
                raise NotImplementedError("'{}' model not yet implemented".format(model))
            
            logger_exp.info("----> [FINISH BUILDING '{}' MODEL FOR '{}' FEATURE SET]".format(model, feature_set))

        logger_exp.info("[FINISH BUILDING MODEL '{}']".format(model))
        
    logger_exp.info("--- FINISH BUILDING MODELS ---")
    return models 

###### train models #######################################################
def train_models(dataset, models):
    logger_exp.info("--- START TRAINING ---")

    for model_key, model_collection in models.items():
        logger_exp.info("[START TRAINING MODEL '{}']".format(model_key))

        for feature_set, model in model_collection.items():
            logger_exp.info("----> [START TRAINING '{}' MODEL FOR '{}' FEATURE SET]".format(model_key, feature_set))
            
            # train model
            X = dataset[feature_set].train_X   
            y = dataset[feature_set].train_y
            model.fit(X, y)

            # show parameter used 
            logger_exp.info("- MODEL PARAMS -")
            logger_exp.info (model.get_params(), pp=True) 

            logger_exp.info("----> [FINISH TRAINING '{}' MODEL FOR '{}' FEATURE SET]".format(model_key, feature_set))
        
        logger_exp.info("[FINISH TRAINING MODEL '{}']".format(model_key))

    logger_exp.info("--- FINISH TRAINING ---")

###### evaluate models ####################################################
def eval_models(args, dataset, models):
    logger_exp.info("--- START EVALUATION ---")
    for model_key, model_collection in models.items():
        logger_exp.info("[START EVALUATING MODEL '{}']".format(model_key))
        for feature_set, model in model_collection.items():
            logger_exp.info("----> [START EVALUATING '{}' MODEL FOR '{}' FEATURE SET]".format(model_key, feature_set))
            
            # evaluate models
            X = dataset[feature_set].test_X
            y = dataset[feature_set].test_y
            metrics = model.evaluate(metrics=args.evaluation_metrics, X=X, y=y)
            
            print(model.get_params())

            # log metrics 
            logger_exp.info("- RESULTS -")
            logger_exp.info(metrics, pp=True)

            logger_exp.info("----> [FINISH EVALUATING '{}' MODEL FOR '{}' FEATURE SET]".format(model_key, feature_set))
        
        logger_exp.info("[FINISH EVALUATING MODEL '{}']".format(model_key))

    logger_exp.info("--- FINISH EVALUATION ---")


###### start experiment (tuning) ##########################################
if __name__ == "__main__":

    # create parser 
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config_file", 
                        default="experiments/1_boosting_models.yaml", 
                        type=argparse.FileType(mode="r"),
                        help="optional, yaml file containing params for experiment")

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
    logger_exp.info("-------- START EXPERIMENT --------")
    logger_exp.info("- PROPERTIES -")
    logger_exp.info (args.__dict__, pp=True) 
    
    try:
        # build dataset 
        dataset = build_dataset(args)
        
        # build models 
        models = build_models(args, dataset)
        
        # train models
        train_models(dataset, models) 
        
        # evaluate model 
        eval_models(args, dataset, models)

        logger_exp.info("-------- FINISH EXPERIMENT --------")
    except:
        e = sys.exc_info()
        logger_exp.exception(e)
   
