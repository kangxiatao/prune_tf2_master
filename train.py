# -*- coding: utf-8 -*-

"""
Created on 06/23/2021
train.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import time
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from utility.log_helper import *
import evaluate
import mycallback
import penalty
import copy


def print_conv2d_norm(trainable_variables, _str='0'):
    # for weight in trainable_variables:
    #     if 'conv' in weight.name and 'kernel' in weight.name:
    #         # 求出其范数值
    #         w_norm = tf.sqrt(tf.reduce_sum(tf.square(weight)))
    #         tf.print(_str+':', w_norm)
    #         break
    # 求出其范数值
    w_norm = tf.sqrt(tf.reduce_sum(tf.square(trainable_variables[0])))
    tf.print(_str+':', w_norm)


class Train:
    def __init__(self, model, args, db_train, db_test, optimizer, lr):
        self.model = model
        self.args = args
        self.db_train = db_train
        self.db_test = db_test
        self.optimizer = optimizer
        self.lr = lr
        self.evaluate = evaluate.Test(model, args)
        self.stop_training = False

        self.save_model = mycallback.ModelSaveToH5(
            model=model,
            filepath=args.save_dir + 'anoi.h5',
            monitor="val_accuracy",
            verbose=1,
        )

        # 用于copy原来的训练参数，外部存储便于构建计算图
        # if self.args.prop_a != 0:
        #     self.oir_trainable_weights = []

    # 描述为静态计算图
    @tf.function
    def train_step(self, x, y):
        _l1_reg = 0
        _l2_reg = 0
        _grouplasso = 0

        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)

            _cost = penalty.cross_entropy_cost(y, logits)

            # Penalty
            if self.args.l1_value != 0.0:
                _l1_reg = penalty.l1_regularization(self.model.trainable_weights, self.args.l1_value)
            if self.args.l2_value != 0.0:
                _l2_reg = penalty.l2_regularization(self.model.trainable_weights, self.args.l2_value)
            if self.args.gl_a != 0.0:
                _grouplasso = penalty.group_lasso(self.model.trainable_weights, self.args.gl_a)

            loss_value = _cost + _l1_reg + _l2_reg + _grouplasso

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        #  Calculation accuracy
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        return loss_value, correct

    # 不便于建图
    # 使用拟合解做group lasso
    # @tf.function
    def train_step_prop(self, x, y):
        # copy一份原来的训练参数
        oir_trainable_weights = []
        for weight in self.model.trainable_variables:
            oir_trainable_weights.append(tf.Variable(weight))
        # print_conv2d_norm(oir_trainable_weights, '0')
        # 计算拟合解
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            _cost = penalty.cross_entropy_cost(y, logits)
        grads = tape.gradient(_cost, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # print_conv2d_norm(oir_trainable_weights, '1')
        # print_conv2d_norm(self.model.trainable_weights, '2')
        # 用拟合解计算group lasso惩罚项，得到新的损失
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            _cost = penalty.cross_entropy_cost(y, logits)
            _grouplasso = penalty.group_lasso(self.model.trainable_weights, self.args.prop_a)
            loss_value = _cost + _grouplasso
        # 回到原来的训练参数
        for i, weight in enumerate(self.model.trainable_variables):
            if 'conv' in weight.name and 'kernel' in weight.name:  # conv2d
                self.model.trainable_variables[i].assign(oir_trainable_weights[i])
        # print_conv2d_norm(self.model.trainable_weights, '3')
        # 用新的损失更新
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # print_conv2d_norm(self.model.trainable_weights, '4')

        #  Calculation accuracy
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        return loss_value, correct

    def __call__(self):
        for epoch in range(self.args.epochs):

            # -- epoch_begin --

            # -- --

            # !!-- epoch_run --!!
            start_time = time.time()
            logging.info('Epoch {}/{}'.format(epoch, self.args.epochs))
            train_loss = 0
            total_correct = 0
            step = 0
            logs = {}

            for step, (x, y) in enumerate(self.db_train):
                # -- batch_begin --
                self.lr.on_batch_begin()
                # -- --

                # !!-- batch_run --!!
                if self.args.prop_a != 0:
                    loss, correct = self.train_step_prop(x, y)
                else:
                    loss, correct = self.train_step(x, y)
                # !!-- --!!

                # -- batch_end --
                self.lr.on_batch_end()
                # -- --

                # loss and acc
                train_loss += loss
                total_correct += int(correct)

            loss = train_loss / (step + 1)
            accuracy = total_correct / self.args.train_set_size

            logging.info('train - {:.1f}s - loss: {:.3f} - accuracy: {:.3f}% ({}/{})'.format(
                time.time() - start_time, loss, 100. * accuracy, total_correct, self.args.train_set_size))

            val_loss, val_accuracy = self.evaluate(self.db_test)

            # get log
            logs['loss'] = loss
            logs['accuracy'] = accuracy
            logs['val_loss'] = val_loss
            logs['val_accuracy'] = val_accuracy

            if self.stop_training:
                break
            # !!-- --!!

            # -- epoch_end --
            self.save_model.on_epoch_end(epoch, logs)
            # -- --
