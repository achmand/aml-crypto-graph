"""
A script which exposes all the datasets used in the experiments. 
The following models are included;
- Elliptic Dataset (https://www.kaggle.com/ellipticco/elliptic-data-set)
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import yaml
from .. import utils as u 
from .elliptic_dr import Elliptic_Dataset
    
###### Datareader functions ###############################################
def get_data(source, config_file="data_config.yaml", **kwargs):
    
    # available sources 
    sources = ["elliptic"]

    # check if source passed is valid 
    if source not in sources:
        error = "'source'=%r is not implemented" % source
        raise NotImplementedError(error)
    
    # load dataset config file .yaml (includes paths to files for a specific dataset)
    with open(config_file, 'r') as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # elliptic dataset (downloadable from: https://www.kaggle.com/ellipticco/elliptic-data-set)
    if source == "elliptic":
        elliptic_data_args = u.Namespace(config["elliptic_dataset"])
        return Elliptic_Dataset(elliptic_data_args, **kwargs)

    # source passed is invalid 
    else:
        error = "source=%r is not implemented" % source
        raise NotImplementedError(error)