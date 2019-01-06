# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas as pd

from xgboost import XGBRegressor

from feature_engineering import feature_engineering
from neural_network import dnn




data_path = 'Data/'

file_name = 'dataset.npz'
model     = 'xgb'
ver       = 0
make_data = True
make_baseline = True




def load(file_name):
    
    a = np.load(file_name)
    
    train_features = a['train_features']
    train_labels = a['train_labels']
    val_features = a['val_features']
    val_labels = a['val_labels']
    test_features = a['test_features']
    feature_list = a['feature_list']

    return train_features, train_labels, val_features, val_labels, test_features, feature_list



def xgb_model(train_features, train_labels, val_features, val_labels):
    
    model = XGBRegressor(
        max_depth=10,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        num_round = 1000,
        eta=0.3,
        seed=42)
    
    
    model.fit(
        train_features,
        train_labels,
        eval_metric="rmse",
        eval_set=[(train_features, train_labels), (val_features, val_labels)],
        verbose=True,
        early_stopping_rounds=10)
    
    return model



def plot_importances(model, feature_list):
    
    importances = list(model.feature_importances_)
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')











if make_data:
    feature_engineering(data_path, file_name)
    

train_features, train_labels, val_features, val_labels, test_features, feature_list = load(file_name)
    

if model == 'xgb' :

    ts = time.time()
    xgb_model = xgb_model(train_features, train_labels, val_features, val_labels)
    tfs, tfm = math.modf((time.time() - ts) / 60)
    print('\n XGB training took {:.0f} minutes {:.0f} seconds' .format(tfm, tfs*60))
    plot_importances(xgb_model, feature_list)
    y_pre = xgb_model.predict(test_features)


elif model == 'dnn' :
    
    ts = time.time()
    y_pre = dnn(train_features, train_labels, val_features, val_labels, test_features)
    y_pre = y_pre[0]
    y_pre = np.reshape(y_pre, (214200))
    tfs, tfm = math.modf((time.time() - ts) / 60)
    print('\n Neural Network training took {:.0f} minutes {:.0f} seconds' .format(tfm, tfs*60))


output = pd.read_csv(str(data_path) + 'sample_submission.csv')
output['item_cnt_month'] = y_pre
output['item_cnt_month'] = output['item_cnt_month'].clip(0, 20)
output[['ID', 'item_cnt_month']].to_csv('%s_ver%d.csv' % (model,ver), index=False)


if make_baseline:
    output = pd.read_csv(str(data_path) + 'sample_submission.csv')
    output['item_cnt_month'] = val_labels
    output['item_cnt_month'] = output['item_cnt_month'].clip(0, 20)
    output[['ID', 'item_cnt_month']].to_csv('baseline.csv', index=False)


 