#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
    @author:huang xiaoqin
    @time: 2018/10/24
    @desc:
"""
"""
the problem of knn is the value of k. if k is too small, the noise will have a higher influence on the result. 
And if k is large, the further one will have an influence on the result. 

There are three key elements of this approach:
(1) a set of labeled objects;
(2) the stored distances between objects;
(3) k, the number of nearest neighbors.

KNN can be used for regression? amazing!
the final step is different. 
KNN做分类预测时，一般是选择多数表决法，即训练集里和预测的样本特征最近的K个样本，预测为里面有最多类别数的类别。
而KNN做回归时，一般是选择平均法，即最近的K个样本的样本输出的平均值作为回归预测值(有一般的平均还有以距离倒数为权重的加权平均)。
"""