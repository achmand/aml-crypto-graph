# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import yaml
import argparse
from logger import Logger


###### start result extraction ############################################
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
    


