#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 03/17/2021
lenet5.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
from tensorflow.keras import Sequential


class LeNet5(tf.keras.Model):
    def __init__(self, class_num=10):
        super(LeNet5, self).__init__()

        self.f = Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

            tf.keras.layers.Conv2D(filters=120, kernel_size=(4, 4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(class_num, activation='relu')
        ])

    def call(self, inputs, training=None, mask=None):

        y = self.f(inputs)

        return y

    @property
    def get_model(self):
        return self.f
