#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
    @author:huang xiaoqin
    @time: 2018/10/25
    @desc:
"""

"""
Logistic regression  gives us a discrete outcome while linear regression gives us a continuous outcome.
That is to say: 逻辑回归（Logistic Regression）是用于处理因变量为分类变量的回归问题,也被称为广义线性回归模型。
重要的是：预测函数和损失函数的形式，具体推导过程可见https://blog.csdn.net/chibangyuxun/article/details/53148005
需要注意的是在直接使用python包时，要注意各种参数的设定。
目标函数实际上是最大似然函数，最大似然的定义是在现象已知的情况下估计参数，所以可以写出在某参数的情况下得到某一事件的概率。
因为现象已知，我们需要使得这一概率最大，所以是最大似然。这里面假设了各个事件是独立的。
https://www.jianshu.com/p/f1d3906e4a3e
"""