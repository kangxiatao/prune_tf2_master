# -*- coding: utf-8 -*-

"""
Created on 04/15/2021
prune.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
import numpy as np
import logging


def get_conv2d(weight):
    _filter = []
    _channel = []
    _data2d = []
    _layer_num = len(weight)
    for i in range(_layer_num):
        layer_data = weight[i]
        # [k, k, c, n]
        _filter.append(np.sum(np.abs(layer_data), axis=(0, 1, 2)))
        _channel.append(np.sum(np.abs(layer_data), axis=(0, 1, 3)))
        _data2d.append(np.sum(np.abs(layer_data), axis=(0, 1)))
    return _filter, _channel, _data2d, _layer_num


def analysis_weight(model):
    _all_conv2d = []
    _variables_num = 0
    # get all conv2d and all variables number
    for weight in model.trainable_variables:
        if 'conv' in weight.name and 'kernel' in weight.name:
            # print(weight.get_shape())
            # print(tf.size(weight))
            _all_conv2d.append(weight)
        if 'conv' in weight.name or 'dense' in weight.name:
            _variables_num += tf.size(weight)

    return get_conv2d(_all_conv2d)


def start_pruning(model, evaluate, db_data, args, mode=None):
    _accuracy, _prune_rate = 0.0, 0.0
    _layer_cnt = 0
    _prune_filter_n = 0
    _last_layer_prune_num = 0
    _rate = 0

    _save_val_acc = evaluate(db_data, verbose=0)[1]
    logging.info('Baseline test accuracy:{}'.format(_save_val_acc))

    # pruning weight
    for i, weight in enumerate(model.trainable_variables):
        # print(weight.name, '---', weight.get_shape())
        if 'conv' in weight.name and 'kernel' in weight.name:  # conv2d
            # print(weight.name, '---', weight.get_shape())
            _shape = weight.get_shape()
            _all_channel_n = _shape[2]
            _all_filter_n = _shape[3]
            _filter = tf.reduce_sum(tf.abs(weight), axis=[0, 1, 2], keepdims=True)
            _layer_cnt += 1

            if mode == 'auto':
                _temp_threshold = 1
                _set_i = 1
                # 这里必须要创建新的张量，不然为引用，在地址上进行修改，导致原来的权重也变了
                _weight_bck = tf.Variable(weight)

                while _temp_threshold > 0.0000001:
                    _filter_bool = tf.greater(_filter, _temp_threshold)  # get prune mark
                    # _filter_bool and prior_prune_bool
                    if args.prior_prune and args.prior_prune_bool_list:
                        _filter_bool = tf.logical_and(_filter_bool, args.prior_prune_bool_list[_layer_cnt-1])
                    _prune_filter_n = _all_filter_n - tf.math.count_nonzero(_filter_bool)
                    # for resnet shortcut2d  这样计算，resnet的剪枝率会略小于实际剪枝率
                    if 'shortcut2d' in weight.name:
                        _rate = (_prune_filter_n / _all_filter_n).numpy()
                    else:
                        _rate = (1 - ((_all_filter_n - _prune_filter_n) * (_all_channel_n - _last_layer_prune_num))
                                 / (_all_channel_n * _all_filter_n)).numpy()
                    # _filter_bool = tf.tile(_filter_bool, [_shape[0], _shape[1], _shape[2], 1])
                    model.trainable_variables[i].assign(tf.where(_filter_bool, weight, tf.zeros_like(weight)))

                    _acc = evaluate(db_data, verbose=0)[1]

                    if _acc >= _save_val_acc:
                        # for resnet shortcut2d  这样计算，resnet的剪枝率会略小于实际剪枝率
                        if 'shortcut2d' not in weight.name:
                            _last_layer_prune_num = _prune_filter_n  # to next layer
                        logging.info('layer: {} | accuracy: {:.5f} | prune rate: {:.5f}({}/{}) | threshold: {}'.format(
                            _layer_cnt, _acc, _rate, _prune_filter_n, _all_filter_n, _temp_threshold))
                        break
                    else:
                        if _set_i % 2 == 1:
                            _temp_threshold /= 2
                        else:
                            _temp_threshold /= 5
                        _set_i += 1
                        weight = tf.Variable(_weight_bck)
                        model.trainable_variables[i].assign(weight)

            else:
                _filter_bool = tf.greater(_filter, args.threshold)  # get prune mark
                # _filter_bool and prior_prune_bool
                if args.prior_prune and args.prior_prune_bool_list:
                    _filter_bool = tf.logical_and(_filter_bool, args.prior_prune_bool_list[_layer_cnt-1])
                _prune_filter_n = _all_filter_n - tf.math.count_nonzero(_filter_bool)
                # for resnet shortcut2d  这样计算，resnet的剪枝率会略小于实际剪枝率
                if 'shortcut2d' in weight.name:
                    _rate = (_prune_filter_n / _all_filter_n).numpy()
                else:
                    _rate = (1 - ((_all_filter_n - _prune_filter_n) * (_all_channel_n - _last_layer_prune_num))
                             / (_all_channel_n * _all_filter_n)).numpy()
                    _last_layer_prune_num = _prune_filter_n  # to next layer
                model.trainable_variables[i].assign(tf.where(_filter_bool, weight, tf.zeros_like(weight)))
                logging.info('layer: {} | prune rate: {:.5f}({}/{}) | threshold: {}'.format(
                    _layer_cnt, _rate, _prune_filter_n, _all_filter_n, args.threshold))

            _prune_rate += _rate

    _accuracy = evaluate(db_data, verbose=0)[1]
    _prune_rate = _prune_rate / _layer_cnt

    return _accuracy, _prune_rate


def prior_pruning(model, evaluate, db_data, args, mode=None):
    _accuracy, _prune_rate = 0.0, 0.0
    _layer_cnt = 0
    _prune_filter_n = 0
    _last_layer_prune_num = 0
    _rate = 0
    _prior_prune_bool_list = []

    _save_val_acc = evaluate(db_data, verbose=0)[1]
    logging.info('Baseline test accuracy:{}'.format(_save_val_acc))

    # pruning weight
    for i, weight in enumerate(model.trainable_variables):
        # print(weight.name, '---', weight.get_shape())
        if 'conv' in weight.name and 'kernel' in weight.name:  # conv2d
            _shape = weight.get_shape()
            _all_channel_n = _shape[2]
            _all_filter_n = _shape[3]
            _filter = tf.reduce_sum(tf.abs(weight), axis=[0, 1, 2], keepdims=True)
            _layer_cnt += 1

            if mode == 'auto':
                _temp_threshold = 1
                _set_i = 1
                # 这里必须要创建新的张量，不然为引用，在地址上进行修改，导致原来的权重也变了
                _weight_bck = tf.Variable(weight)

                while _temp_threshold > 0.0000001:
                    _filter_bool = tf.greater(_filter, _temp_threshold)  # get prune mark
                    _prune_filter_n = _all_filter_n - tf.math.count_nonzero(_filter_bool)
                    # for resnet shortcut2d  这样计算，resnet的剪枝率会略小于实际剪枝率
                    if 'shortcut2d' in weight.name:
                        _rate = (_prune_filter_n / _all_filter_n).numpy()
                    else:
                        _rate = (1 - ((_all_filter_n - _prune_filter_n) * (_all_channel_n - _last_layer_prune_num))
                                 / (_all_channel_n * _all_filter_n)).numpy()
                    # _filter_bool = tf.tile(_filter_bool, [_shape[0], _shape[1], _shape[2], 1])
                    model.trainable_variables[i].assign(tf.where(_filter_bool, weight, tf.zeros_like(weight)))

                    _acc = evaluate(db_data, verbose=0)[1]

                    if _acc >= _save_val_acc:
                        _prior_prune_bool_list.append(_filter_bool)
                        # for resnet shortcut2d  这样计算，resnet的剪枝率会略小于实际剪枝率
                        if 'shortcut2d' not in weight.name:
                            _last_layer_prune_num = _prune_filter_n  # to next layer
                        logging.info('layer: {} | accuracy: {:.5f} | prune rate: {:.5f}({}/{}) | threshold: {}'.format(
                            _layer_cnt, _acc, _rate, _prune_filter_n, _all_filter_n, _temp_threshold))
                        break
                    else:
                        if _set_i % 2 == 1:
                            _temp_threshold /= 2
                        else:
                            _temp_threshold /= 5
                        _set_i += 1
                        weight = tf.Variable(_weight_bck)
                        model.trainable_variables[i].assign(weight)

            else:
                _filter_bool = tf.greater(_filter, args.threshold)  # get prune mark
                _prior_prune_bool_list.append(_filter_bool)
                _prune_filter_n = _all_filter_n - tf.math.count_nonzero(_filter_bool)
                # for resnet shortcut2d  这样计算，resnet的剪枝率会略小于实际剪枝率
                if 'shortcut2d' in weight.name:
                    _rate = (_prune_filter_n / _all_filter_n).numpy()
                else:
                    _rate = (1 - ((_all_filter_n - _prune_filter_n) * (_all_channel_n - _last_layer_prune_num))
                             / (_all_channel_n * _all_filter_n)).numpy()
                    _last_layer_prune_num = _prune_filter_n  # to next layer
                model.trainable_variables[i].assign(tf.where(_filter_bool, weight, tf.zeros_like(weight)))
                logging.info('layer: {} | prune rate: {:.5f}({}/{}) | threshold: {}'.format(
                    _layer_cnt, _rate, _prune_filter_n, _all_filter_n, args.threshold))

            _prune_rate += _rate

    _accuracy = evaluate(db_data, verbose=0)[1]
    _prune_rate = _prune_rate / _layer_cnt

    return _accuracy, _prune_rate, _prior_prune_bool_list






