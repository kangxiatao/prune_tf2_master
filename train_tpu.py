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
import evaluate_tpu
import mycallback
import penalty


class Train:
    def __init__(self, model, strategy, args, db_train, db_test, optimizer, lr):
        self.model = model
        self.strategy = strategy
        self.args = args
        self.db_train = db_train
        self.db_test = db_test
        self.optimizer = optimizer
        # self.loss = penalty.GoGoGoLoss(model, args)
        self.lr = lr
        self.evaluate = evaluate_tpu.Test(model, strategy, args)
        self.stop_training = False

        self.save_model = mycallback.ModelSaveToH5(
            model=model,
            filepath=args.save_dir + 'anoi.h5',
            monitor="val_accuracy",
            verbose=1,
        )

    # 描述为静态计算图
    @tf.function
    def train_step(self, x, y):

        def step_fn(images, labels):
            _l1_reg = 0
            _l2_reg = 0
            _grouplasso = 0

            with tf.GradientTape() as tape:
                logits = self.model(images, training=True)

                _cost = penalty.cross_entropy_cost(labels, logits)

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
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.reduce_sum(tf.cast(correct, tf.float32))
            return loss_value, correct

        return self.strategy.run(step_fn, args=(x, y))

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
                loss, correct = self.train_step(x, y)
                # !!-- --!!

                # -- batch_end --
                self.lr.on_batch_end()
                # -- --

                # loss and acc
                train_loss += float(sum(loss.values))
                total_correct += int(sum(correct.values))

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
