# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from tqdm import tqdm
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math


def feature_engineering(data_path, output_filename):
    
    
    ts = time.time()
    dataset = data_pre_processing(data_path)
    
    dataset['date_block_num']   = dataset['date_block_num'].astype(np.int8)
    dataset['shop_id']          = dataset['shop_id'].astype(np.int8)
    dataset['item_id']          = dataset['item_id'].astype(np.int16)
    dataset['city_id']          = dataset['city_id'].astype(np.int8)
    dataset['item_category_id'] = dataset['item_category_id'].astype(np.int8)
    dataset['item_cnt_month']   = dataset['item_cnt_month'].astype(np.int8)
     
    
    
    print('\n STEP 6/7 ADD NEW FEATURES')
    dataset, new_features = add_features(dataset)
    feature_list = ['date_block_num', 'shop_id', 'city_id', 'item_id', 'item_category_id'] +  new_features + ['item_cnt_month']
    dataset = dataset[feature_list]
    dataset = dataset[dataset['date_block_num'] > 11]
    
    
    
    print('\n STEP 7/7 DATA PREPARATION FOR MACHINE LEARNING')
    
    dataset['item_cnt_month'] = dataset.item_cnt_month.fillna(0).clip(0,20)

    dataset_lo_table = (dataset['date_block_num'] >= 0)
    dataset_hi_table = (dataset['date_block_num'] <= 32)
    validation_table = (dataset['date_block_num'] == 33)
    test_table       = (dataset['date_block_num'] == 34)

    trainset_ml        = dataset[dataset_lo_table & dataset_hi_table]
    validationset_ml   = dataset[validation_table]
    testset_ml         = dataset[test_table]

    trainset_ml_values = trainset_ml.values.astype(int)
    train_features     = trainset_ml_values[:, 0:dataset.shape[1] - 1]
    train_labels       = trainset_ml_values[:, dataset.shape[1] - 1]
    
    validationset_ml_values = validationset_ml.values.astype(int)
    val_features            = validationset_ml_values[:, 0:dataset.shape[1] - 1]
    val_labels              = validationset_ml_values[:, dataset.shape[1] - 1]

    testset_ml_values = testset_ml.values.astype(int)
    test_features     = testset_ml_values[:, 0:dataset.shape[1] - 1]
    
    np.savez(output_filename, train_features = train_features, val_features = val_features, train_labels = train_labels, val_labels = val_labels, test_features = test_features, feature_list = feature_list)
    
    tfs, tfm = math.modf((time.time() - ts) / 60)
    print('\n Data pre-processing took {:.0f} minutes {:.0f} seconds' .format(tfm, tfs*60))



def data_pre_processing(data_path):
    
    print('\n STEP 1/7 LOADING DATA')
    categories, items, sales, shops, sales_test = load_data(data_path)
    grid = []
    for block_num in sales['date_block_num'].unique():
        x = sales_test['shop_id'].unique()
        y = sales_test['item_id'].unique()    
        grid.append(np.array(list(product(*[x,y,[block_num]])),dtype='int32'))
        
    index_cols = ['shop_id', 'item_id', 'date_block_num']
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

    
    print('\n STEP 2/7 REMOVING DUPLICATES')
    delete_duplicates('sales', sales, sub_set=['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day'])
    delete_duplicates('sales_test',  sales_test,  sub_set=['shop_id', 'item_id'])
    
    print('\n STEP 3/7 ANOMALY DETECTION')
    anomaly_detection('sales_train__date_block_num', sales['date_block_num'])
    anomaly_detection('sales_train__item_price',     sales['item_price'])
    anomaly_detection('sales_train__item_cnt_day',   sales['item_cnt_day'])

    print('\n STEP 4/7 CORRECTING ANOMALIES')
    median = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) & (sales.item_price > 0)].item_price.median()
    sales.loc[sales.item_price < 0, 'item_price'] = median
    sales['item_price'] = sales['item_price'].clip(0, 300000)
    sales['item_cnt_day'] = sales['item_cnt_day'].clip(0, 1000)
        
    sales, sales_test, grid, shops, categories = fix_cats(sales, sales_test, grid, shops, categories, items)   

    print('\n STEP 5/7 SUM MONTH SALES')
    sales['item_cnt_day'] = sales['item_cnt_day'].clip(0,20)
    groups = sales.groupby(['shop_id', 'city_id', 'item_id', 'item_category_id', 'date_block_num'])
    trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
    trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
    trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)  
    
    sales_test['date_block_num'] = 34
    sales_test['item_cnt_month'] = -1
    testset = sales_test
        
    trainset = trainset [['date_block_num', 'shop_id', 'city_id', 'item_id', 'item_category_id', 'item_cnt_month']]
    testset  = testset  [['date_block_num', 'shop_id', 'city_id', 'item_id', 'item_category_id', 'item_cnt_month']]
    
    trainset = pd.merge(grid, trainset, how='left').fillna(0)
    dataset = pd.concat([trainset, testset], axis = 0) 

    return dataset




def load_data(data_path):
        
    categories  = pd.read_csv(data_path + 'item_categories.csv')
    items       = pd.read_csv(data_path + 'items.csv')
    sales       = pd.read_csv(data_path + 'sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
    shops       = pd.read_csv(data_path + 'shops.csv')
    sales_test  = pd.read_csv(data_path + 'test.csv').set_index('ID')
    
    alphabet_change(categories, 'item_category_name')
    alphabet_change(items, 'item_name')
    alphabet_change(shops, 'shop_name')
       
    return categories, items, sales, shops, sales_test



def alphabet_change(data_frame, column_to_translate):
    
    symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
               u"abvgdeejzijklmnoprstufhzcssyyyeuaABVGDEEJZIJKLMNOPRSTUFHZCSSYYYEUA")

    tr = {ord(a):ord(b) for a, b in zip(*symbols)}

    def convert(russian):
        english=russian.translate(tr)
        return english

    data_frame[column_to_translate] = data_frame.apply(lambda row: convert(row[column_to_translate]), axis=1)
  
    
def delete_duplicates(name, data_frame, sub_set):
      
    before = data_frame.shape[0]
    data_frame.drop_duplicates(sub_set, keep='first', inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    after = data_frame.shape[0]
    print('{} duplicates: {}'.format(name, before - after))

def anomaly_detection (name, series):
    
    print('\n anomaly detection for {}'.format(name))
    print(series.describe())
    
    plt.plot(series)
    plt.title(name)
    plt.show()

def fix_cats(sales, sales_test, grid, shops, categories, items):
    
    
    #### FIX AND ADD SHOPS ####
    
    shops.loc[shops.shop_name == '!Akutsk Ordjonikidze, 56 fran', 'shop_name'] = 'Akutsk Ordjonikidze, 56'
    shops.loc[shops.shop_name == '!Akutsk TZ "Zentralynyj" fran', 'shop_name'] = 'Akutsk TZ "Zentralynyj"'
    shops.loc[shops.shop_name == 'Jukovskij ul. Ckalova 39m?', 'shop_name'] = 'Jukovskij ul. Ckalova 39m²'
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    
    fix_sales = pd.merge(sales, shops, how='left')
    fix_test  = pd.merge(sales_test, shops, how='left')
    fix_grid  = pd.merge(grid, shops, how='left')
    
    shops = shops.drop([11,57,58])
    shops['shop_id'] = LabelEncoder().fit_transform(shops['shop_id'])
    shops = shops.reset_index()
    shops = shops.drop(['index'], axis=1)
    shops.rename(columns = {'shop_id':'shop_id_fix'}, inplace = True)
    
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])       ## add info about city
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_id'] = LabelEncoder().fit_transform(shops['city'])

    
    sales = pd.merge(fix_sales, shops, how='left')
    sales_test = pd.merge(fix_test, shops, how='left')
    grid  = pd.merge(fix_grid, shops, how='left')
    
    shops = shops[['shop_id_fix','city_id']]
    shops.rename(columns = {'shop_id_fix':'shop_id'}, inplace = True)
    

    #### FIX AND ADD CATS ####

    l_cat = list(categories.item_category_name)
    for ind in range(0,1):
        l_cat[ind] = 'PC Headsets / Headphones'
    for ind in range(1,8):
        l_cat[ind] = 'Access'
    l_cat[8] = 'Tickets (figure)'
    l_cat[9] = 'Delivery of goods'
    for ind in range(10,18):
        l_cat[ind] = 'Consoles'
    for ind in range(18,25):
        l_cat[ind] = 'Consoles Games'
    l_cat[25] = 'Accessories for games'
    for ind in range(26,28):
        l_cat[ind] = 'phone games'
    for ind in range(28,32):
        l_cat[ind] = 'CD games'
    for ind in range(32,37):
        l_cat[ind] = 'Card'
    for ind in range(37,43):
        l_cat[ind] = 'Movie'
    for ind in range(43,55):
        l_cat[ind] = 'Books'
    for ind in range(55,61):
        l_cat[ind] = 'Music'
    for ind in range(61,73):
        l_cat[ind] = 'Gifts'
    for ind in range(73,79):
        l_cat[ind] = 'Soft'
    for ind in range(79,81):
        l_cat[ind] = 'Office'
    for ind in range(81,83):
        l_cat[ind] = 'Clean'
    l_cat[83] = 'Elements of a food'

    lb = preprocessing.LabelEncoder()
    categories['item_category_id_fix'] = lb.fit_transform(l_cat)
    categories['item_category_name_fix'] = l_cat
    
    sales = pd.merge(sales, items, how='left')
    sales_test = pd.merge(sales_test, items, how='left')
    grid  = pd.merge(grid, items, how='left')
    
    sales = sales.merge(categories[['item_category_id', 'item_category_id_fix']], on = 'item_category_id', how = 'left')
    _ = sales.drop(['item_category_id'],axis=1, inplace=True)
    sales.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)
    
    sales_test = sales_test.merge(categories[['item_category_id', 'item_category_id_fix']], on = 'item_category_id', how = 'left')
    _ = sales_test.drop(['item_category_id'],axis=1, inplace=True)
    sales_test.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)
    
    grid = grid.merge(categories[['item_category_id', 'item_category_id_fix']], on = 'item_category_id', how = 'left')
    _ = grid.drop(['item_category_id'],axis=1, inplace=True)
    grid.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)
    
    _ = categories.drop(['item_category_id'],axis=1, inplace=True)
    _ = categories.drop(['item_category_name'],axis=1, inplace=True)

    categories.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)
    categories.rename(columns = {'item_category_name_fix':'item_category_name'}, inplace = True)
    categories = categories.drop_duplicates()
    categories.index = np.arange(0, len(categories))
    

    ### DELETE COLUMNS WITH NAMES ### 
    
    sales = sales[['date', 'date_block_num', 'shop_id_fix', 'city_id', 'item_id', 'item_category_id', 'item_price', 'item_cnt_day']]
    sales_test = sales_test [['shop_id_fix', 'city_id', 'item_id', 'item_category_id']]
    grid  = grid [['date_block_num', 'shop_id_fix', 'city_id', 'item_id', 'item_category_id']]
    
    sales.rename(columns = {'shop_id_fix':'shop_id'}, inplace = True)
    sales_test.rename(columns = {'shop_id_fix':'shop_id'}, inplace = True)
    grid.rename(columns = {'shop_id_fix':'shop_id'}, inplace = True)
    
    return sales, sales_test, grid, shops, categories  



def add_features(dataset):
    
    new_features = []
    lookback_range = [1,2,3,6,12]
    use_feature = [True, True, True, True, False, False]  #item/shop #item/city #item #cat/shop #cat/city #cat
    

    tqdm.pandas()
    
    if use_feature[0]:
        print('PREVIOUS ITEM SALES in SHOP')
        for diff in tqdm(lookback_range):
            feature_name = 'item/shop -' + str(diff)
            dataset2 = dataset.copy()
            dataset2.loc[:, 'date_block_num'] += diff
            dataset2.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            dataset = dataset.merge(dataset2[['shop_id', 'item_id', 'date_block_num', feature_name]], on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
            dataset[feature_name] = dataset[feature_name].fillna(0).clip(0,20).astype(np.int8)
            new_features.append(feature_name)
    
    if use_feature[1]:
        print('PREVIOUS ITEM SALES in CITY')
        groups = dataset.groupby(by = ['city_id', 'item_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'item/city -' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            dataset = dataset.merge(result, on = ['city_id', 'item_id', 'date_block_num'], how = 'left')
            dataset[feature_name] = dataset[feature_name].fillna(0).clip(0,20).astype(np.int8)
            new_features.append(feature_name) 
        
    if use_feature[2]:
        print('PREVIOUS ITEM SALES')
        groups = dataset.groupby(by = ['item_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'item -' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            dataset = dataset.merge(result, on = ['item_id', 'date_block_num'], how = 'left')
            dataset[feature_name] = dataset[feature_name].fillna(0).clip(0,20).astype(np.int8)
            new_features.append(feature_name) 
    
    if use_feature[3]:
        print('PREVIOUS CATS SALES in SHOP')
        groups = dataset.groupby(by = ['shop_id', 'item_category_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'cat/shop -' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            dataset = dataset.merge(result, on = ['shop_id', 'item_category_id', 'date_block_num'], how = 'left')
            dataset[feature_name] = dataset[feature_name].fillna(0).clip(0,20).astype(np.int8)
            new_features.append(feature_name) 
        
    if use_feature[4]:
        print('PREVIOUS CATS SALES in CITY')
        groups = dataset.groupby(by = ['city_id', 'item_category_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'cat/city -' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            dataset = dataset.merge(result, on = ['city_id', 'item_category_id', 'date_block_num'], how = 'left')
            dataset[feature_name] = dataset[feature_name].fillna(0).clip(0,20).astype(np.int8)
            new_features.append(feature_name) 
    
    if use_feature[5]:
        print('PREVIOUS CATS SALES')
        groups = dataset.groupby(by = ['item_category_id', 'date_block_num'])
        for diff in tqdm(lookback_range):
            feature_name = 'cat -' + str(diff)
            result = groups.agg({'item_cnt_month':'mean'})
            result = result.reset_index()
            result.loc[:, 'date_block_num'] += diff
            result.rename(columns={'item_cnt_month': feature_name}, inplace=True)
            dataset = dataset.merge(result, on = ['item_category_id', 'date_block_num'], how = 'left')
            dataset[feature_name] = dataset[feature_name].fillna(0).clip(0,20).astype(np.int8)
            new_features.append(feature_name) 
    

    return dataset, new_features