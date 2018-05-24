# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import pandas as pd
import numpy as np
# 检查统计部分 列表
# 可以单独对数据进行分析和检测
def continuous_check(value ,check_cache ,check_name):
    '''检查数据是否连续,也就是说 数据是否需要根据uid 进行排序'''
    current_uid = value[0]
    last_uid = check_cache['last_uid'] if 'last_uid' in check_cache.keys() else None
    if 'tmp' not in check_cache.keys():
        check_cache['tmp'] = {}
    tiny_cache = check_cache['tmp']
    old_set = tiny_cache['oid_set'] if 'oid_set' in tiny_cache.keys() else set()
    if last_uid != current_uid:
        '''直接判断当前current_uid 是否有被缓存过'''
        if current_uid in old_set:
            check_cache[check_name] = 0
        if last_uid != None:
            old_set.add(last_uid)
        check_cache['tmp']['oid_set'] = old_set
    check_cache['last_uid'] = current_uid
    if check_name not in check_cache:
        check_cache[check_name] = 1
    return check_cache

def inc_check(value ,check_cache ,check_name):
    current_uid = int(value[0])
    last_uid = int(check_cache['last_uid']) if 'last_uid' in check_cache.keys() else -1
    if current_uid < last_uid:
        check_cache[check_name] = 0
    if check_name not in check_cache:
        check_cache[check_name] = 1
    return check_cache


# 判断时间是否连续增长
def time_inc_check(item,check_cache,check_name):
    is_inc =1 if  ((item[1] - item[1].shift(1))[1:] > 0).all() else 0
    if (check_name in check_cache):
        if check_cache[check_name] == 1:
            check_cache[check_name] = is_inc
    else:
        check_cache[check_name] =  1
    return check_cache

# 用户最大行程和最小行程数量
def trip_stats(item,check_cache,check_name):
    item_trip_num = len(item[2].unique())
    item_trip_min = item[2].min()
    max_name = check_name+'_max'
    min_name = check_name+'_min'
    sum_name = check_name + '_sum'
    mins_name = check_name + '_first'
    max_value = check_cache.get(max_name,0)
    min_value = check_cache.get(min_name,np.inf)
    sum_value = check_cache.get(sum_name,0)
    mins_trip = check_cache.get(mins_name,set())
    if item_trip_num > max_value:
        check_cache[max_name] = item_trip_num
    if item_trip_num < min_value:
        check_cache[min_name] =item_trip_num
    ctotal = sum_value + item_trip_num
    check_cache[sum_name] = ctotal
    mins_trip.add(item_trip_min)
    check_cache[mins_name] = mins_trip
    return check_cache




check_value_dict = {u'数据ID是否聚集' :continuous_check ,u'数据ID是否连续增长' :inc_check}
check_item_dict = {u'判断时间是否连续增长':time_inc_check,u'用户行程统计':trip_stats}
