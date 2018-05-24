# -*- coding: utf-8 -*-
# from sklearn import datasets
# import lightgbm as lgb
# # boston = datasets.load_boston()
# #
# # data_boston = boston['data']
# # label_boston = boston['target']
# #
# # lgbranker = lgb.LGBMRanker()
# # lgbranker.fit(data_boston,label_boston)

import pandas as pd
import numpy as np

def name_recover(key_value_d):
    ret_set = set()
    for key,value in key_value_d.items():
        key_list = key.split('_')
        f_name = '_'.join(key_list[:-1])
        ret_set.add(f_name)
    return list(ret_set)

def columns_add(df, feature_name):
    item_series = df[feature_name[0]]
    for item in feature_name[1:]:
        item_series = item_series + df[item]
    return item_series

def columns_sub(ini,nf,feature_name):
    item_series = ini.copy()
    for item in feature_name:
        item_series = item_series - nf[item]
    return item_series

class F1Rank:
    '''test 级别的model'''
    def __init__(self):
        pass

    def fit(self,train,Y):
        return

    def predict(self,test):
        return

    def builder_rank(self,test,cache_dict_train):
        '''随机生成排序'''
        f_min = cache_dict_train['analysis_min']
        f_max = cache_dict_train['analysis_max']
        pos_feats,neg_feats = name_recover(f_max),name_recover(f_min)

        pf = np.log(test[pos_feats].rank())
        nf = np.log(test[neg_feats].rank())

        ret = columns_add(pf,pos_feats)
        ret_v = columns_sub(ret,nf,neg_feats)
        return ret_v