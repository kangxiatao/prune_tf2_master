# -*- coding: utf-8 -*-

"""
Created on 03/19/2021
tf_test.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import os
import tensorflow as tf
import cv2
import math
import numpy as np
from utility.loaddata import *

# use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# weight = tf.ones([3, 3, 16, 6])
weight = tf.random.normal(shape=[3, 3, 16, 6], mean=0, stddev=1, dtype=tf.float32)
pi = tf.constant(math.pi)

# 这里先求出其范数值，然后按范数值大小得到索引，取前60%
channel_norm = tf.sqrt(tf.reduce_sum(tf.square(weight), axis=(0, 1, 3)))
filter_norm = tf.sqrt(tf.reduce_sum(tf.square(weight), axis=(0, 1, 2)))

channel_norm_index = tf.argsort(channel_norm, direction='DESCENDING')
channel_norm_index = channel_norm_index[:int(channel_norm_index.shape[0] * 0.6)]  # 取前60%
print(channel_norm_index)
channel_weight = tf.gather(weight, channel_norm_index, axis=2)
filter_norm_index = tf.argsort(filter_norm, direction='DESCENDING')
filter_norm_index = filter_norm_index[:int(filter_norm_index.shape[0] * 0.6)]  # 取前60%
filter_weight = tf.gather(weight, filter_norm_index, axis=3)

# 得到性质与weight一样的均值向量tensor
one_weight = tf.ones_like(channel_weight)
channel_mean = tf.reduce_mean(channel_weight, axis=(0, 1, 2))  # 通道的均值  个数等于过滤器数
channel_mean = one_weight * channel_mean
one_weight = tf.ones_like(filter_weight)
filter_mean = tf.reduce_mean(filter_weight, axis=(0, 1, 3))  # 过滤器的均值  个数等于通道数
one_weight = tf.transpose(one_weight, (0, 1, 3, 2))
filter_mean = one_weight * filter_mean
filter_mean = tf.transpose(filter_mean, (0, 1, 3, 2))

# 重新计算范数值，也就是模的大小
channel_norm = tf.sqrt(tf.reduce_sum(tf.square(channel_weight), axis=(0, 1, 3)))  # 模的大小
channel_mean_norm = tf.sqrt(tf.reduce_sum(tf.square(channel_mean), axis=(0, 1, 3)))
filter_norm = tf.sqrt(tf.reduce_sum(tf.square(filter_weight), axis=(0, 1, 2)))
filter_mean_norm = tf.sqrt(tf.reduce_sum(tf.square(filter_mean), axis=(0, 1, 2)))
# 向量与均值向量点积
channel_dot_mean = tf.reduce_sum(channel_weight * channel_mean, axis=(0, 1, 3))
filter_dot_mean = tf.reduce_sum(filter_weight * filter_mean, axis=(0, 1, 2))

# 求出向量到均值向量的角余弦值
cos_channel = tf.divide(channel_dot_mean, channel_norm * channel_mean_norm + tf.constant(0.0000001))
cos_filter = tf.divide(filter_dot_mean, filter_norm * filter_mean_norm + tf.constant(0.0000001))
# 余弦相似度 取值范围为[0, 1]
similarity_channel = 1 - tf.divide(tf.acos(cos_channel), pi)
similarity_filter = 1 - tf.divide(tf.acos(cos_filter), pi)

cos = tf.reduce_sum(similarity_filter) * 1
# cos = tf.reduce_sum(similarity_channel) * 1

print(cos)

# data_filename = "../../Data/cifar-10-python/cifar-10-batches-py"
#
#
# raw_data = Cifar(data_filename)
# train_x, train_y, test_x, test_y = raw_data.prepare_data()
#
# print(train_x.shape)
#
# for i in range(255):
#     img = test_x[i]
#     # img = cv2.resize(img, (32, 32))
#     cv2.namedWindow("cs")                                           # 创建一个窗口，名称cs
#     cv2.imshow("cs", img)                                            # 在窗口cs中显示图片
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()                                         # 释放窗口
#
#     # print(type(img))
#     # paddings = tf.constant([[4, 4], [4, 4]])
#     padding = 4
#     npad = ((padding, padding), (padding, padding), (0, 0))
#     img = tf.pad(img, npad)  # padding
#     img = tf.image.random_crop(img, [32, 32, 3])
#     # print("000000000000000000")
#     # print(type(img.numpy()))
#     cv2.imshow("cs", img.numpy())                                            # 在窗口cs中显示图片
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()                                         # 释放窗口
