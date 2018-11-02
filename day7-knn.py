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

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv('datasets/Social_Network_Ads.csv')
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# p=2 means euclidean_distance
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import precision_score,accuracy_score
# 预测正确的正样本／所有预测为正的样本数（因此参数的前后不能写错了）
precision = precision_score(y_test, y_pred)
print 'precision:', precision
# 预测正确的样本数／总样本数 （参数前后颠倒计算出来的值不变，但是还是按要求写比较好）
accuracy = accuracy_score(y_test, y_pred)
print 'accuracy:', accuracy