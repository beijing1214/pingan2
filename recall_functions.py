# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''from cavities main function '''

import pandas as pd
import numpy as np
from .feature_functions import feature_func, dtypes_train, dtypes_test, tranfrom_itemgeter
from .check_functions import check_value_dict, check_item_dict
import csv
import json
import datetime


# 时间跨度 约184天，
# 训练集：33454 个用户， 23,734,760个记录
# 测试集： 14045 个用户， 9,914,734 个记录


def check_value_function(value, check_cache, check_dict=check_value_dict):
    current = check_cache
    for check_name, check_item_func in check_dict.items():
        current = check_item_func(value, current, check_name)
    return current


def check_item_function(item, check_cache, check_dict=check_item_dict):
    current = check_cache
    for check_name, check_item_func in check_dict.items():
        current = check_item_func(item, current, check_name)
    return current


def ext_function(dataitem, check_cache, skip_func, check_item_func,dtype_dict, func_list=feature_func,debug=False):
    '''特征工程具体函数'''
    is_skip = skip_func(dataitem)
    if is_skip:
        return pd.DataFrame(), is_skip, check_cache
    item = pd.DataFrame(dataitem)
    item = tranfrom_itemgeter(item, dtype_dict)
    check_cache = check_item_func(item, check_cache)
    data_collection = {}
    for func_name, func in func_list.items():
        item_feature = func(item)
        if debug:
            print(item_feature)
            continue
        if type(item_feature) == dict:
            data_collection.update(item_feature)
        else:
            data_collection[func_name] = item_feature
    return pd.Series(data_collection), is_skip, check_cache


def label_skip(item):
    if len(item) == 0:
        return False
    if float(item[0][9]) <= 0:
        return True
    else:
        return False


def skip_func(item, skip_list=None):
    ret = False
    if skip_list == None:
        return False
    for skip_func in skip_list:
        current = skip_func(item)
        if current == True:
            return True
    return ret


# csv_file        文件名字
# target_field    userId字段所在
# do_function     特征提取方法
# label           Y 获取
# check_item_function  检查数据是否有效方法
def online_prepare(csv_file, target_field=0, do_function=ext_function, label=9,
                   check_value_function=check_value_function, skip_function=skip_func,
                   check_global_function=check_item_function,debug = False,dtype_dict=(dtypes_train, dtypes_test)):
    fp = open(csv_file, 'r')
    target_id = None
    csv_iter = csv.reader(fp)
    current_package = []
    features = pd.DataFrame()
    userid = []
    Y = []
    dd = dtype_dict[0]
    if label == None:
        dd = dtype_dict[1]


    '''data info'''

    check_cache = {}
    check_cache['start_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    user_num = 0
    count_num = 0
    for i, value in enumerate(csv_iter):
        if i == 0:
            continue
        if check_value_function != None:
            '''检查函数'''
            check_cache = check_value_function(value, check_cache)
        if (target_id == None) or (target_id == value[target_field]):
            current_package.append(value)
            target_id = value[target_field]
            if label != None:
                current_y = value[label]
        else:
            item, skip_flag, check_cache = do_function(current_package, check_cache, skip_function,
                                                       check_global_function,debug=debug,dtype_dict=dd
                                                       )
            if debug:
                return userid, features, Y, check_cache
            if skip_flag:
                target_id = value[target_field]
                current_package = []
                continue
            count_num += len(current_package)
            user_num += 1
            if label != None:
                Y.append(current_y)
            try:
                features = features.append(item, ignore_index=True)
            except Exception as e:
                print('what?\t' + item)
            userid.append(target_id)
            current_package = []
            current_package.append(value)
            target_id = value[target_field]
    # 最后一段数据处理
    item, skip_flag, check_cache = do_function(current_package, check_cache, skip_function, check_global_function,debug=debug,dtype_dict=dd)
    if skip_flag:
        fp.close()
        if check_value_function != None:
            del check_cache['tmp']
            check_cache['mean_count'] = (i + 0.0) / (len(userid) + 0.0) if len(userid) != 0 else 0.0
            check_cache['count_num'] = count_num
            check_cache['user_num'] = user_num
            # print(parse.urlencode(check_cache).encode('utf-8'))
            # print(HTTPResponse(check_cache))
            # print(json.dumps(check_cache, encoding="UTF-8", ensure_ascii=False))
        return userid, features, Y, check_cache
    try:
        features = features.append(item, ignore_index=True)
    except Exception as e:
        print('what?\t' + item)
    target_id = value[target_field]
    userid.append(target_id)
    if label != None:
        Y.append(value[label])
    fp.close()
    if check_value_function != None:
        del check_cache['tmp']
        check_cache['mean_count'] = i / (len(userid) + 0.0)
        count_num += len(current_package)
        check_cache['count_num'] = count_num
        check_cache['user_num'] = user_num + 1
        check_cache['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        # print(parse.urlencode(check_cache).encode('utf-8'))
        # , encoding="UTF-8", ensure_ascii=False)

        # print(json.dumps(check_cache, encoding="UTF-8", ensure_ascii=False))
    Y = np.array(Y, dtype=float)
    userid = np.array(userid, dtype=int)
    return userid, features, Y, check_cache

# -------------------------------------------------------------------------------
# demo_path = 'E:/python-workdir/data/PINGAN-2018-train_demo.csv'
# userid, features, Y = online_prepare(demo_path)
# print(userid)
# print(features)
# print(Y)
