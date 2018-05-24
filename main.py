# -*- coding: utf-8 -*-
'''from cavities main function '''
from __future__ import absolute_import, division, print_function

import os
import sys
import datetime

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import numpy as np
from df_online_info.recall_functions import online_prepare
from df_online_info.model_functions import *
from df_online_info.features_analysis import feature_analysis,kendalltau_coef,spearmanr_coef,pearsonr_coef,gini_coef
from df_online_info.model_functions import F1Rank

import pandas as pd

def save_rank(userid,rank_value,path):
    ret =  pd.DataFrame({'Id':userid,'Pred':rank_value})
    ret.to_csv(path,index=False,line_terminator='\n')




train_path = '/data/dm/train.csv'
pred_path = '/data/dm/test.csv'
demo_path = 'E:/python-workdir/data/PINGAN-2018-train_demo.csv'
save_path = 'model/result_'

if __name__ == '__main__':
    # 69306
    stat_time = datetime.datetime.now()
    ''' 添加feature_functions 内容 统计出特征'''
    userid_train, features_train, Y_train, cache_dict_train = online_prepare(train_path)

    userid_test, features_test, Y_test, cache_dict_test = online_prepare(pred_path,label=None)

    '''对结果和特征进行统计分析---start'''
    f_max,f_min = feature_analysis(features_train, Y_train)
    cache_dict_train['analysis_max'] = f_max
    cache_dict_train['analysis_min'] = f_min
    cache_dict_train['feature_len'] = features_train.shape
    print('{0}'.format(cache_dict_train))
    '''对结果和特征进行统计分析---end  可以不做 以提高效率'''


    '''来源于model function 可以自己弄一个'''
    model = F1Rank()
    model.fit(features_train,Y_train)

    '''随机生成排序的规则后期可以进行保存'''
    rank_value = model.builder_rank(features_test,cache_dict_train)
    VERSION = stat_time.strftime('%Y%m%d%H%M')
    save_rank(userid_test,rank_value,save_path+VERSION+'.csv')


    end_time = datetime.datetime.now()
    value = end_time-stat_time
    print('cost time = {0}'.format(value))
'''
25   1  1476923280   6  122.978073  41.095844   31  43.192444   2.790000  0   
26   1  1476923340   6  122.979309  41.098061   22  48.085327   8.120000  0   
27   1  1476923400   6  122.981155  41.099125   47  44.475220   5.100000  0   
28   1  1476923460   6  122.983727  41.101570   34  45.226379   3.510000  0   
29   1  1476923520   6  122.984138  41.101902   40  49.462158   4.180000  0   
'''

