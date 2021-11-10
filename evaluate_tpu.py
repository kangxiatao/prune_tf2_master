# -*- coding: utf-8 -*-

"""
Created on 06/24/2021
evaluate.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import time
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from utility.log_helper import *
import penalty


class Test:
    def __init__(self, model, strategy, args):
        self.model = model
        self.strategy = strategy
        self.args = args
        # self.loss = penalty.GoGoGoLoss(model, args)

    # 描述为静态计算图
    @tf.function
    def test_step(self, x, y):

        def step_fn(images, labels):
            # loss_value = tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
            # loss_value = tf.nn.compute_average_loss(loss_value, global_batch_size=self.args.batch_size)

            logits = self.model(images)

            loss_value = penalty.cross_entropy_cost(labels, logits)

            # Penalty
            if self.args.l1_value != 0.0:
                loss_value += penalty.l1_regularization(self.model.trainable_weights, self.args.l1_value)
            if self.args.l2_value != 0.0:
                loss_value += penalty.l2_regularization(self.model.trainable_weights, self.args.l2_value)
            if self.args.gl_a != 0.0:
                loss_value += penalty.group_lasso(self.model.trainable_weights, self.args.gl_a)
            if self.args.prop_a != 0.0:
                loss_value += penalty.group_lasso(self.model.trainable_weights, self.args.prop_a)

            #  Calculation accuracy
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.reduce_sum(tf.cast(correct, tf.float32))
            return loss_value, correct

        return self.strategy.run(step_fn, args=(x, y))

    def __call__(self, db_test, verbose=1):

        test_loss = 0
        total_correct = 0
        step = 0

        for step, (x, y) in enumerate(db_test):
            # -- batch_begin --

            # -- --

            # -- batch_run --
            loss, correct = self.test_step(x, y)
            # -- --

            # -- batch_end --

            # -- --

            # loss and acc
            test_loss += float(sum(loss.values))
            total_correct += int(sum(correct.values))

        val_loss = test_loss / (step + 1)
        val_accuracy = total_correct / self.args.test_set_size

        if verbose > 0:
            logging.info('evaluate - val_loss: {:.3f} - val_accuracy: {:.3f}% ({}/{})'.format(
                val_loss, 100. * val_accuracy, total_correct, self.args.test_set_size))

        return val_loss, val_accuracy
