# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from geopy.distance import vincenty
from geopy.geocoders import Nominatim

u''' 特征工程部分 函数'''
dtypes_train = {
    0: 'uint32',
    1: 'uint32',
    2: 'float32',
    3: 'float32',
    4: 'float32',
    5: 'uint8',
    6: 'float32',
    7: 'float32',
    8: 'uint8',
    9: 'float32'
}

dtypes_test = {
    0: 'uint32',
    1: 'uint32',
    2: 'float32',
    3: 'float32',
    4: 'float32',
    5: 'uint8',
    6: 'float32',
    7: 'float32',
    8: 'uint8'
}


def base_feature(item_np,feature_name):
    v = {}
    v[feature_name + '_mean'] = item_np.mean()
    v[feature_name + '_std'] = item_np.std()
    v[feature_name + '_skew'] = item_np.skew()
    v[feature_name + '_kurt'] = item_np.kurt()
    v[feature_name + '_min'] = item_np.min()
    v[feature_name + '_max'] = item_np.max()
    v[feature_name + '_median'] = item_np.median()
    v[feature_name + '_q25'] = item_np.quantile(0.25)
    v[feature_name + '_q75'] = item_np.quantile(0.75)
    v[feature_name + '_var'] = item_np.var()
    return v


def base_feature_extract(df,columns):
    ret = {}
    for col in columns:
        i_ret =base_feature(df[col],col)
        ret.update(i_ret)
    return ret



geolocator = Nominatim()





def dtype_columns(dtype_dict):
    ret = {}
    for i,v in dtype_dict.items():
        c = ret.get(v,[])
        c.append(i)
        ret[v] = c
    return ret

# dtypes_train = dtype_columns(dtypes_train)
# dtypes_test =dtype_columns(dtypes_test)

def tranfrom_itemgeter(item,dtypes_dict,sorted_list=[1,2]):

    ret = item.astype(dtypes_dict,copy=True)

    # for type,columns in dtypes_dict.items():
    #     item[columns] = item[columns].astype(type)

    ret.sort_values(sorted_list,inplace=True)
    return ret



def callstate(item):
    '''通话记录'''
    callstate_init = {0:0,1:0,2:0,3:0,4:0}
    uniq,count = np.unique(item[8],return_counts=True)
    sumcount = count/(sum(count)+0.0)
    s = dict(zip(uniq,sumcount))
    callstate_init.update(s)
    ret = {'callstate_{0}'.format(k):v for k,v in callstate_init.items()}
    return ret

def geo_info(item):
    '''时间间隔'''
    time_gap =(item[1:][1] - item.shift(1)[1:][1])
    end_local = item[1:][[3,4,7]]
    start_local = item.shift(1)[1:][[3,4,7]]
    start_local.columns = ['start_longitude','start_latitude','start_speed']
    end_local.columns = ['end_longitude','end_latitude','end_speed']
    loacl_se =pd.concat([start_local,end_local,time_gap],axis=1)
    loacl_se['distance_se'] = loacl_se.apply(lambda x: vincenty((x['start_latitude'],x['start_longitude']),(x['end_latitude'],x['end_longitude'])).m,axis=1)
    loacl_se['distance_speed'] = loacl_se['distance_se']/loacl_se[1]*3600/1000
    # 平均加速度
    loacl_se['avg_acc'] = (loacl_se['start_speed'] - loacl_se['end_speed'])/loacl_se[1]
    loacl_se['sub_start_acc'] = (loacl_se['start_speed'] - loacl_se['distance_speed'])/loacl_se[1]/2
    loacl_se['sub_end_acc'] = (loacl_se['end_speed'] - loacl_se['distance_speed']) / loacl_se[1]/ 2
    # 绝对值
    loacl_se['avg_acc_l2'] = np.sqrt(np.abs((loacl_se['start_speed']**2 - loacl_se['end_speed']**2)/loacl_se[1]))
    loacl_se['sub_start_acc_l2'] = np.sqrt(np.abs(loacl_se['start_speed']**2 - loacl_se['distance_speed']**2)/loacl_se[1]/2)
    loacl_se['sub_end_acc_l2'] = np.sqrt(np.abs(loacl_se['end_speed'] - loacl_se['distance_speed']) / loacl_se[1]/ 2)
    # 时间超过60s
    # 二阶特征
    # loacl_se['sub_time_ada'] = (loacl_se[1] > 60).cumsum()


    # 构造一阶特征
    # 速度类
    speed_columns = ['distance_speed','avg_acc','sub_start_acc','sub_end_acc','avg_acc_l2', 'sub_start_acc_l2','sub_end_acc_l2']
    res = base_feature_extract(loacl_se,speed_columns)
    return res


'''要处理的函数 注册进 feature_func'''
'''注册方法 {'方法名字'：方法的统计特征}  '''
feature_func = {u'callstate_stats': callstate,'geo_speed':geo_info}
