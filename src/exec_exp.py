# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### Importing dependencies ############################################# 
import yaml
import argparse
from cryptoaml.utils import Namespace

# Datasets
import cryptoaml.datareader as cdr

###### Load dataset #######################################################
def build_dataset(args):
    if args.data == "elliptic":
        elliptic_args = Namespace(args.elliptic_args)
        elliptic_data = cdr.get_data(source="elliptic", 
                                     config_file=args.data_config_file,
                                     encode_classes=elliptic_args.encode_classes)
        dataset = elliptic_data.train_test_split(train_size=elliptic_args.train_size,
                                                 feat_set=elliptic_args.feat_sets,
                                                 inc_meta=False,
                                                 inc_unknown=elliptic_args.inc_unknown)
        return dataset
    else:
        raise NotImplementedError("'{}' not yet implemented".format(args.data))

###### Start experiment ###################################################
if __name__ == "__main__":
    
    # Create parser 
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config_file", 
                        default="experiments/1_boosting_models.yaml", 
                        type=argparse.FileType(mode="r"),
                        help="optional, yaml file containing params for experiment")

    # Parse arguments from config file 
    args = parser.parse_args()
    if args.config_file:
        properties = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, "config_file")
        arg_dict = args.__dict__
        for key, value in properties.items():
            arg_dict[key] = value

    # Build dataset 
    dataset = build_dataset(args)

    # Build models 
    