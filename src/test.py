import time
from collections import Counter

import numpy
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, \
    precision_recall_curve
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
import numpy as np

def get_dataset(filename):
    import pandas as pd

    csv_file = filename
    df = pd.read_csv(csv_file)

    Y = df['FLAG']
    X = df.loc[:, df.columns != 'FLAG']
    X.pop('Index')
    X.pop('ERC20_most_sent_token_type')
    X.pop('ERC20_most_rec_token_type')
    X.pop('ERC20_uniq_sent_token_name')
    X.pop('ERC20_uniq_rec_token_name')
    #X.pop('Address')

    X.fillna(0, inplace=True)
    return X, Y

if __name__ == '__main__':
    X, Y = get_dataset('data/eth_fraud/eth_accounts.csv')

    tmp_columns = X.columns
    print(tmp_columns)
    print(len(tmp_columns))

    #k_means(X,Y)
    #X_diff, Y_diff = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/new_illicit_addresses.csv')
    #X_diff, Y_diff = get_dataset('C:/Users/luter/Documents/Github/Ethereum_Fraud_Detection/Account_Stats/Complete_Illicit_Subset_1000.csv')
    # stratified_k_fold_XGBoost(X, Y, n_folds=10)

    # importance_list = []
    # num_of_train_test_splits = 10
    # for i in range(num_of_train_test_splits):
    #     X_train, X_test, y_train, y_test = prepare_dataset_split(X, Y, testSize=0.1)
    #     #XGBoost(X_train, y_train, X_diff, Y_diff) # TO TEST NEWLY ADDED ADDRESSES
    #     #XGBoost(X_train, y_train, X_test, y_test)
    #     importance = XGBoost(X_train, y_train, X_test, y_test)
    #     importance_list.extend(importance)

    # sorted_feature_list = update_list(importance_list)
    #plot_average_importance_values(sorted_feature_list, num_of_train_test_splits)

    #   random_forest(X_train,  y_train, X_test, y_test)

