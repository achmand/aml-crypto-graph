"""

"""

###### importing dependencies #############################################
import pandas as pd
from .. import utils as u 
from collections import OrderedDict
from ._datareader import _BaseDatareader

###### noaa weather dataset ###############################################
class Weather_Dataset(_BaseDatareader):

    # constants -----------------------------------------------------------
    # columns - meta 
    COL_TS     = "ts"    # timestep 
    # columns - features 
    COL_TEMP   = "temperature"
    COL_DW     = "dew_point"
    COL_SEA    = "sea_level_pressure"
    COL_VIS    = "visibility"
    COL_WIND   = "avg_wind_speed"
    COL_SWIN   = "max_sustained_wind_speed"
    COL_MXTEMP = "max_temperature"
    COL_MITEMP = "min_temperature"

    # labels 
    LABEL_RAIN   = "no_rain"   
    LABEL_NORAIN = "rain"

    # constructor ---------------------------------------------------------
    def __init__(self, 
                 data_args):
        # call parent constructor 
        super().__init__(data_args=data_args)

    # properties ----------------------------------------------------------
    @property
    def labels_(self):
        """Gets the label and it's respective encoding."""
        return {self.LABEL_NORAIN:  0, 
                self.LABEL_RAIN: 1}

    # load data functions -------------------------------------------------
    def _load_dataset(self, data_args):

        # 1. load features and classes 
        features_path = "{}{}".format(data_args.folder, data_args.features_file)
        labels_path   = "{}{}".format(data_args.folder, data_args.classes_file)
        features_df   = pd.read_csv(features_path, header=None)
        labels_df     = pd.read_csv(labels_path, header=None)

        # 2. set column names 
        self._cols_features = [self.COL_TEMP, 
                               self.COL_DW, 
                               self.COL_SEA, 
                               self.COL_VIS, 
                               self.COL_WIND, 
                               self.COL_SWIN, 
                               self.COL_MXTEMP, 
                               self.COL_MITEMP]
        
        if data_args.processed == True:
            self._cols_features = self._cols_features 
            features_df.columns = self._cols_features + [self.COL_TS]

        # 2. add timestep column (every 30 records)
        if data_args.processed == False:
            features_df.columns = self._cols_features
            features_df[self.COL_TS] = (features_df.index / 30) + 1
            features_df[self.COL_TS] = features_df[self.COL_TS].astype(int) 
            
        # 3. merge features and labels 
        features_df[self.COL_CLASS] = labels_df[0]

        # 4. encode labels to binary 
        if data_args.processed == False:
            features_df[self.COL_CLASS] = features_df[self.COL_CLASS].apply(
                lambda x: 1 if x == 2 else 0)  

        return features_df

    def train_test_split(self, train_size, as_dict=True):

        # we need to find out the timestamp we are splitting with given train_size
        last_ts = self._dataset.tail(1)[self.COL_TS]
        split_ts = int(last_ts * train_size)

        # split training and test by timestamp (temporal splitting)
        data_train = self._dataset[self._dataset[self.COL_TS] <= split_ts]
        data_test = self._dataset[self._dataset[self.COL_TS] > split_ts]

        # create input split 
        data_train_X = data_train[self._cols_features + [self.COL_TS]]
        data_test_X = data_test[self._cols_features + [self.COL_TS]]

        # create output split 
        data_train_y = data_train[self.COL_CLASS]
        data_test_y = data_test[self.COL_CLASS]

        # create dictionary to hold dataset 
        datasplit = {}
        datasplit["train_X"] = data_train_X
        datasplit["train_y"] = data_train_y
        datasplit["test_X"] = data_test_X
        datasplit["test_y"] = data_test_y

        # make as object
        datasplit = u.Namespace(datasplit) 

        # check if it should be returned as dictionary or dataplit
        if as_dict == True:
            data = OrderedDict()
            data["ALL"] = datasplit
            return data
        else:
            return datasplit

