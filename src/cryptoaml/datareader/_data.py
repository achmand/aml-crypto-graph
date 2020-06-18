"""
A script which exposes all the datasets used in the experiments. 
The following datasets are included;
- Elliptic Dataset (https://www.kaggle.com/ellipticco/elliptic-data-set)
- Ethereum Fraud Detection (https://github.com/sfarrugia15/Ethereum_Fraud_Detection) 
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import yaml
from .. import utils as u 
from ._noaa_dr import Weather_Dataset   
from ._elliptic_dr import Elliptic_Dataset
from ._eth_accounts_dr import Eth_Accounts_Dataset   

###### Datareader functions ###############################################
def get_data(source, config_file="configuration/data/data_config.yaml", **kwargs):
    
    # available sources 
    sources = ["elliptic", "eth_accounts", "noaa_weather"]

    # check if source passed is valid 
    if source not in sources:
        error = "'source'=%r is not implemented" % source
        raise NotImplementedError(error)
    
    # load dataset config file .yaml (includes paths to files for a specific dataset)
    with open(config_file, "r") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # elliptic dataset (downloadable from: https://www.kaggle.com/ellipticco/elliptic-data-set)
    if source == "elliptic":
        elliptic_data_args = u.Namespace(config["elliptic_dataset"])
        return Elliptic_Dataset(elliptic_data_args, **kwargs)

    # eth fraud accounts dataset (downloadable from: https://github.com/sfarrugia15/Ethereum_Fraud_Detection/blob/master/Account_Stats/Complete.csv)
    elif source == "eth_accounts":
        eth_accounts_args = u.Namespace(config["eth_accounts_dataset"])
        return Eth_Accounts_Dataset(eth_accounts_args, **kwargs)

    # noaa weather dataset (downloadable from: http://users.rowan.edu/~polikar/res/nse/weather_data.zip)
    elif source == "noaa_weather":
        noaa_weather_args = u.Namespace(config["noaa_weather_dataset"])
        return Weather_Dataset(noaa_weather_args, **kwargs)

    # source passed is invalid 
    else:
        error = "source=%r is not implemented" % source
        raise NotImplementedError(error)