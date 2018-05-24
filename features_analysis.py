# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import numpy as np
import pandas as pd
from scipy.stats import kendalltau,spearmanr, pearsonr
from df_online_info.eval_function import gini, gini_normalized

'''一些评价指标'''


def pearsonr_coef(feats, Y, prefix_name='_pearsonr'):
    f = feats.copy()
    item_col_name = f.columns
    item_col_name = item_col_name + prefix_name
    f.columns = item_col_name
    ret = f.apply(lambda x:pearsonr(x,Y)[0],axis=0)
    ret_dict =  ret.to_dict()
    return ret_dict,min(ret_dict,key=ret_dict.get),max(ret_dict, key=ret_dict.get)


def spearmanr_coef(feats, Y, prefix_name='_spearmanr'):
    f = feats.copy()
    item_col_name = f.columns
    item_col_name = item_col_name + prefix_name
    f.columns = item_col_name
    ret = f.apply(lambda x: pearsonr(x, Y)[0], axis=0)
    ret_dict =  ret.to_dict()
    return ret_dict,min(ret_dict,key=ret_dict.get),max(ret_dict, key=ret_dict.get)

def kendalltau_coef(feats, Y, prefix_name='_kendalltau'):
    f = feats.copy()
    item_col_name = f.columns
    item_col_name = item_col_name + prefix_name
    f.columns = item_col_name
    ret = f.apply(lambda x: kendalltau(x, Y)[0], axis=0)
    ret_dict =  ret.to_dict()
    return ret_dict,min(ret_dict,key=ret_dict.get),max(ret_dict, key=ret_dict.get)


def gini_coef(feats,Y,prefix_name='_gini'):
    f = feats.copy()
    item_col_name = f.columns
    item_col_name = item_col_name + prefix_name
    f.columns = item_col_name
    ret = f.apply(lambda x: gini(x, Y), axis=0)
    ret_dict = ret.to_dict()
    return ret_dict, min(ret_dict, key=ret_dict.get),max(ret_dict, key=ret_dict.get)

analysis_list = [pearsonr_coef,spearmanr_coef,kendalltau_coef,gini_coef]

'''特征分析的方法 线性相关 gini 以及 排序相关性'''
def feature_analysis(feats, Y, analysis_func=analysis_list):
    ret = {}
    f_max_d = {}
    f_min_d = {}
    for func in analysis_func:
        item_dict,f_min,f_max = func(feats, Y)
        ret.update(item_dict)
        f_max_d[f_max] = ret[f_max]
        f_min_d[f_min] = ret[f_min]

    return f_max_d,f_min_d
