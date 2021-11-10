# -*- coding: utf-8 -*-

"""
Created on 03/20/2021
penalty.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
import math
import numpy as np
from utility.log_helper import *


@tf.function
def cross_entropy_cost(y_true, y_pred):
    _cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    return _cost


def l1_regularization(trainable_variables, args):
    loss_regularization = []
    for p in trainable_variables:
        if 'conv' in p.name and 'kernel' in p.name:
            loss_regularization.append(tf.math.reduce_sum(tf.math.abs(p)))
    return args * tf.reduce_sum(tf.stack(loss_regularization))


def l2_regularization(trainable_variables, args):
    loss_regularization = []
    for p in trainable_variables:
        if 'conv' in p.name and 'kernel' in p.name:
            # loss_regularization.append(tf.nn.l2_loss(p))
            loss_regularization.append(tf.sqrt(tf.math.reduce_sum(tf.math.square(p))))
    return args * tf.reduce_sum(tf.stack(loss_regularization))


def group_lasso(trainable_variables, args):
    loss_gl = []
    for weight in trainable_variables:
        if 'conv' in weight.name and 'kernel' in weight.name:
            tt = tf.reduce_sum(tf.abs(weight), axis=[0, 1])
            tt = tf.math.square(tt)
            t1 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tt, axis=1)))
            t2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tt, axis=0)))
            loss_gl.append(t1 * args + t2 * args)
    return tf.reduce_sum(tf.stack(loss_gl))


# 似乎loss用一个类包装起来不好操控，分开求好些
class GoGoGoLoss:
    def __init__(self, model, args):
        self.model = model  # tf中是把model作可变数据类型
        self.args = args
        self.last_w = []  # 上一次迭代的权重

    # @tf.function
    def __call__(self, y_true, y_pred):
        _l1_reg = 0
        _l2_reg = 0
        _var_reg = 0
        _grouplasso = 0
        _prop_reg = 0

        _cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        # _cost = tf.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        # _cost = tf.nn.compute_average_loss(_cost, global_batch_size=self.args.batch_size)

        if self.args.l1_value != 0.0:
            loss_regularization = []
            for p in self.model.trainable_variables:
                if 'conv' in p.name and 'kernel' in p.name:
                    loss_regularization.append(tf.math.reduce_sum(tf.math.abs(p)))
            _l1_reg = self.args.l1_value * tf.reduce_sum(tf.stack(loss_regularization))

        if self.args.l2_value != 0.0:
            loss_regularization = []
            for p in self.model.trainable_variables:
                if 'conv' in p.name and 'kernel' in p.name:
                    loss_regularization.append(tf.nn.l2_loss(p))
                    # loss_regularization.append(tf.math.reduce_sum(tf.math.square(p)))
            _l2_reg = self.args.l2_value * tf.reduce_sum(tf.stack(loss_regularization))

        if self.args.var_1 != 0.0 or self.args.var_2 != 0.0:
            cos = tf.constant(0.0)
            pi = tf.constant(math.pi)
            for weight in self.model.trainable_variables:
                if 'conv' in weight.name and 'kernel' in weight.name:
                    # 这里先求出其范数值，然后按范数值大小得到索引，取前x%
                    channel_norm = tf.sqrt(tf.reduce_sum(tf.square(weight), axis=(0, 1, 3)))
                    filter_norm = tf.sqrt(tf.reduce_sum(tf.square(weight), axis=(0, 1, 2)))

                    channel_norm_index = tf.argsort(channel_norm, direction='DESCENDING')
                    channel_norm_index = channel_norm_index[
                                         :int(channel_norm_index.shape[0] * self.args.penalty_ratio)]  # 取前x%
                    channel_weight = tf.gather(weight, channel_norm_index, axis=2)
                    filter_norm_index = tf.argsort(filter_norm, direction='DESCENDING')
                    filter_norm_index = filter_norm_index[
                                        :int(filter_norm_index.shape[0] * self.args.penalty_ratio)]  # 取前x%
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

                    cos += tf.reduce_sum(similarity_filter) * self.args.var_1
                    cos += tf.reduce_sum(similarity_channel) * self.args.var_2
            _var_reg = cos

        if self.args.gl_1 != 0.0 or self.args.gl_2 != 0.0:
            loss_gl = []
            for weight in self.model.trainable_variables:
                if 'conv' in weight.name and 'kernel' in weight.name:
                    t1 = tf.reduce_sum(tf.abs(weight), axis=[0, 1, 2])
                    t2 = tf.reduce_sum(tf.abs(weight), axis=[0, 1, 3])
                    t1 = t1 * t1
                    t2 = t2 * t2
                    t1 = tf.sqrt(tf.reduce_sum(t1))
                    t2 = tf.sqrt(tf.reduce_sum(t2))
                    loss_gl.append(t1 * self.args.gl_1 + t2 * self.args.gl_2)
            _grouplasso = tf.reduce_sum(tf.stack(loss_gl))

        if self.args.gl_a != 0.0:
            loss_gl = []
            for weight in self.model.trainable_variables:
                if 'conv' in weight.name and 'kernel' in weight.name:
                    tt = tf.reduce_sum(tf.abs(weight), axis=[0, 1])
                    tt = tf.math.square(tt)
                    t1 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tt, axis=1)))
                    t2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tt, axis=0)))
                    loss_gl.append(t1 * self.args.gl_a + t2 * self.args.gl_a)
            _grouplasso = tf.reduce_sum(tf.stack(loss_gl))

        # proportional compression
        """
            计算核和组的范数增长比例的比例（核比组），根据比例调节惩罚力度，共两个超参数（暂时就用一个）
            需要保存上次的核和组范数值，我是直接保存上次的权重
        """
        if self.args.prop_1 != 0.0 or self.args.prop_2 != 0.0:
            loss_gl = []
            _later_cnt = 0
            _prop_var = 1
            for weight in self.model.trainable_variables:
                if 'conv' in weight.name and 'kernel' in weight.name:
                    # 分为过滤器和通道两个组
                    ww_t1 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(weight), axis=[0, 1, 2]))))
                    ww_t2 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(weight), axis=[0, 1, 3]))))
                    # 计算惩罚力度
                    if len(self.last_w):
                        _last_w = self.last_w[_later_cnt]
                        ww_tt = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(weight), axis=[0, 1]))))
                        ll_tt = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(_last_w), axis=[0, 1]))))
                        w2d_prop = tf.divide(ww_tt, ll_tt)
                        ll_t1 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(_last_w), axis=[0, 1, 2]))))
                        ll_t2 = tf.sqrt(tf.reduce_sum(tf.math.square(tf.reduce_sum(tf.abs(_last_w), axis=[0, 1, 3]))))
                        gro_prop = (tf.divide(ww_t1, ll_t1) + tf.divide(ww_t2, ll_t2)) / 2
                        _prop_var = w2d_prop / gro_prop
                        tf.print(w2d_prop, gro_prop, _prop_var)
                    # 第一次不做惩罚力度控制
                    else:
                        pass

                    loss_gl.append((ww_t1 + ww_t2) * self.args.prop_1 * _prop_var)

                    # 层计数
                    _later_cnt += 1

            # 保存新的权重
            self.last_w.clear()
            for weight in self.model.trainable_variables:
                if 'conv' in weight.name and 'kernel' in weight.name:
                    self.last_w.append(weight)

            _prop_reg = tf.reduce_sum(tf.stack(loss_gl))

        """
        以下部分暂且失败，没找到好的方法，如要实现需重构代码
        # # proportional compression
        # if self.args.prop_1 != 0.0 or self.args.prop_2 != 0.0:
        #     oir_mode1 = self.model
        #     l2_mode1 = self.model
        # 
        #     oir_mode1.fit(self.db_train, validation_data=self.db_test, epochs=1, verbose=0, callbacks=self.callbacks)
        #     l2_mode1.fit(self.db_train, validation_data=self.db_test, epochs=1, verbose=0, callbacks=self.callbacks)
        # 
        #     oir_norm = []
        #     l2_norm = []
        #     for weight in oir_mode1.trainable_variables:
        #         if 'conv' in weight.name and 'kernel' in weight.name:
        #             t1 = tf.sqrt(tf.reduce_sum(tf.abs(weight)))
        #             oir_norm.append(t1)
        #     for weight in l2_mode1.trainable_variables:
        #         if 'conv' in weight.name and 'kernel' in weight.name:
        #             t1 = tf.sqrt(tf.reduce_sum(tf.abs(weight)))
        #             l2_norm.append(t1)
        # 
        #     proportion = list(map(lambda x: x[1] / x[0], zip(oir_norm, l2_norm)))
        #     proportion = np.mean(proportion)
        #     print(proportion)
        # 
        #     return _prop_reg
        # else:
        #     return _cost + _l1_reg + _l2_reg + _var_reg + _grouplasso
        """

        return _cost + _l1_reg + _l2_reg + _var_reg + _grouplasso + _prop_reg
