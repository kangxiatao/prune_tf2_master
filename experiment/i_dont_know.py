import numpy as np
import tensorflow as tf
import math

# train_labels = [0, 2, 1, 4, 9]
# train_labels = np.array(train_labels)
# train_labels[train_labels > 1] = 1
# print(train_labels)
#
# train_labels = np.eye(2)[train_labels]
# print(train_labels)

# weight = tf.ones([3, 3, 16, 6])
# _last_w = tf.ones([3, 3, 16, 6])
# weight = tf.ones([3, 3])
# weight = tf.random.normal(shape=[3, 3, 16, 6], mean=0, stddev=1, dtype=tf.float32)
# _last_w = tf.random.normal(shape=[3, 3, 16, 6], mean=0, stddev=1, dtype=tf.float32)
# pi = tf.constant(math.pi)

# print(tf.math.abs(weight).shape)
# print(tf.math.reduce_sum(tf.math.abs(weight)).shape)
# print(tf.math.reduce_sum(tf.math.abs(weight)))
#
# print(tf.squeeze(tf.math.square(weight)).shape)
# print(tf.math.square(weight))
# print(tf.squeeze(tf.math.reduce_sum(tf.math.square(weight))).shape)
# print(tf.math.reduce_sum(tf.math.square(weight)))
#
# print(tf.nn.l2_loss(weight).shape)
# print(tf.nn.l2_loss(weight))

# ww_tt = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(weight), axis=[0, 1]))))
# ll_tt = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(_last_w), axis=[0, 1]))))
# w2d_prop = tf.divide(ww_tt, ll_tt)
# ww_t1 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(weight), axis=[0, 1, 2]))))
# ww_t2 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(weight), axis=[0, 1, 3]))))
# ll_t1 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(_last_w), axis=[0, 1, 2]))))
# ll_t2 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(_last_w), axis=[0, 1, 3]))))
# gro_prop = (tf.divide(ww_t1, ll_t1) + tf.divide(ww_t2, ll_t2)) / 2
# _prop_var = w2d_prop / gro_prop
#
# tf.print(w2d_prop, gro_prop, _prop_var)

# l = [1, 2, 3]
#
#
# class GoGoGoLoss:
#     def __init__(self, model, args):
#         self.model = model
#         self.args = args
#         self.last_w = []  # 上一次迭代的权重
#
#     def __call__(self, x):
#         self.model = self.model + x
#
#
# aa = GoGoGoLoss(l, 1)
# # aa([4])
# l += [4]
# print(aa.model)
# print(l)


rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
print(float(rank_0_tensor))

aa = (1,2,3)

print(sum(aa))