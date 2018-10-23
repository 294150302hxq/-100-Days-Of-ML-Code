#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
    @author:huang xiaoqin
    @time: 2018/10/23
    @desc:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('datasets/Social_Network_Ads.csv')
X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 0均值标准化，（x-u）/sigma
# why this is important? 1.for training speed 2.for precision, because the calculation of distance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 这个时候sc会根据X_train得到一个缩放工具并进行缩放
X_train = sc.fit_transform(X_train)
# 进行缩放
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
# confusion matrix contains TP,FP,FN,TN
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score,accuracy_score
# 预测正确的正样本／所有预测为正的样本数（因此参数的前后不能写错了）
precision = precision_score(y_test, y_pred)
print 'precision:', precision
# 预测正确的样本数／总样本数 （参数前后颠倒计算出来的值不变，但是还是按要求写比较好）
accuracy = accuracy_score(y_test, y_pred)
print 'accuracy:', accuracy

