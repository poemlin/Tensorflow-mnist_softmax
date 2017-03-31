#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:28:13 2017

@author: mml
"""
# tensorflow的MNIST数据加载模块
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 载入tensorflow
import tensorflow as tf
# 将session注册为默认session
sess = tf.InteractiveSession()

# placeholder定义输入数据的地方
# 数据类型；[个数，数据维数]，none表示不限
x = tf.placeholder(tf.float32,[None,784])
# variable存储模型参数
# Variable长期存在并且每轮更新
# 初始参数设置为0
#（注意初始参数在很多情况至关重要，这里由于模型简单才设为0）
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# softmax函数
# softmax是tf.nn下的函数，tf.nn包含大量神经网络组件
# tf.matmul是矩阵乘法函数
y = tf.nn.softmax(tf.matmul(x,w)+b)

# y_存储真实label
y_ = tf.placeholder(tf.float32,[None,10])
# 定义损失函数
# reduce_sum求和，reduce_mean对每一个batch数据求均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

# 只要定义好优化目标（损失函数）和优化算法，tensorflow就会自动求导进行反向传播
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 全局参数初始化器
tf.initialize_all_variables().run()
# 前面只是定义了优化目标和优化算法，batch里必须feed数据才能进行训练
# 定义迭代次数和batch数据
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    
#测试
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})