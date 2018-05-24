# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import kendalltau

''' 模型训练的 eval function'''
def dcg_score(pred, label, k=5):
    order = np.argsort(pred)[::-1]
    y_true = np.take(label, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

# label ---- > [0,1,0,1,0,1]
def idcg_score(pred, label):
    order = np.argsort(pred)[::-1]
    y_true = np.take(label, order)
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

# 排序 rank score
def ndcg_score(pred, label, k=None):
    if k == None:
        dcg_max = idcg_score(label, label)
        dcg_min = idcg_score(label, -label)
        assert dcg_max > dcg_min
        if not dcg_max:
            return 0.
        dcg = idcg_score(pred, label)
        return (dcg - dcg_min) / (dcg_max - dcg_min)
    dcg_max = dcg_score(label,label,k)
    dcg_min = dcg_score(label,-label,k)
    assert dcg_max > dcg_min
    if not dcg_max:
        return 0.
    dcg = dcg_score(pred,label ,k)
    return (dcg - dcg_min) / (dcg_max - dcg_min)


# gini dataset init
def gini(pred,actual):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    total_losses = all[:, 0].sum()
    gini_sum = all[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)

def gini_normalized(pred,actual):
    max_gini = gini(actual,actual)
    min_gini = gini(actual,-actual)
    assert  max_gini>min_gini
    return (gini(actual, pred) - min_gini)/(max_gini-min_gini)


def inc_eprem(sorted_value,premium,reparations):
    order = np.argsort(sorted_value)
    premium_sorted = np.take(premium, order)
    reparations_sorted = np.take(reparations,order)
    premium_cumsum = np.cumsum(premium_sorted)
    reparations_cumsum = np.cumsum(reparations_sorted)
    premium_sum = np.sum(premium)
    reparations_sum = np.sum(reparations)
    x = premium_cumsum/premium_sum
    y = reparations_cumsum/reparations_sum
    return x,y




def pingan_gini(sorted_value,premium,reparations):
    x,y = inc_eprem(sorted_value,premium,reparations)
    ret = gini(x,y)
    return ret

def idcg_mse(sorted_value,f):
    order = np.argsort(sorted_value)[::-1]
    f_order = np.take(f,order)
    f_rank = np.sort(f)[::-1]
    ret1 = np.sum((f_order - f_rank)**2/len(f_order))
    ret2 = np.sqrt(ret1)
    return ret2

def argsort_mse(sorted_value,f):
    order = np.argsort(sorted_value)[::-1]
    f_order = np.argsort(f)[::-1]
    ret = np.mean((np.abs(f_order - order)+0.0)  / len(f_order))
    return ret




#  # 指标分析  最好能够预测出赔付金额 用来排序 其次用赔付率
# np.random.seed(10021)
# # 数据生成
# n = 220
# p = 0.13
# k = np.arange(0,1000)
#
# x = binom.pmf(k,n,p)
# x = np.random.rand(1000)
# # # print(x)
# np.random.shuffle(x)
# y =  np.random.rand(1000)
# z = [np.random.rand(1000) for i in range(10000)]
# f = x/y
#
# print(x.shape,y.shape)

# # k = [ np.abs(np.log(i)) if i !=0 else 0 for i in f]
# # k1 = [ -np.log(i) if i !=0 else 0 for i in f]
# # k2 = [ np.abs(np.log(i)) if i !=0 else -1 for i in f]
# print(pingan_gini(f,x,y))
# print(pingan_gini(k,x,y))
# # print(pingan_gini(k1,x,y))
# # print(pingan_gini(x,x,y))
# # print(pingan_gini(y,x,y))
# # print(gini(x,x))
# # print(gini(f,f))
# # print(gini(x,x))
#
#
#
#
# #
# #
# # k = [ np.abs(np.log(i)) if i !=0 else 0 for i in f]
# #
# gini_rank = [pingan_gini(i,x,y)  for i in z]
# #
# # # #
# # # gini_i=[gini(i,f)  for i in z]
# # print(gini(k,f))
# # #
# dcg_rank =  [kendalltau(i,f)[0]  for i in z]
# import matplotlib.pyplot as plt
# plt.scatter(gini_rank,dcg_rank)
# # plt.scatter(gini_rank,f)
# plt.show()
# pred = np.array([1,2,3,4,5])
# label = np.array([4,2,3,5,1])
#
# print(ndcg_score(-pred,pred))