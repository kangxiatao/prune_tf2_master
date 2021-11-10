#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 03/17/2021
vgg16.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import math
import cv2
import matplotlib.pyplot as plt


def viz_layer(data, layer=0, n_filters=16, n_im=0):
    fig = plt.figure(figsize=(20, 20))
    len_f = data[layer].shape[3]
    for i in range(0, len_f, int(len_f/n_filters)):
        ax = fig.add_subplot(math.sqrt(n_filters), math.sqrt(n_filters), i + 1, xticks=[], yticks=[])
        ax.imshow(data[layer][n_im, :, :, i], cmap='gray')
        ax.set_title('Output %s' % str(i + 1))


class VGG16(tf.keras.Model):
    def __init__(self, class_num=10):
        super(VGG16, self).__init__()
        self.conv1 = Sequential([layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu')
                                 ])
        self.conv2 = Sequential([layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
                                 ])
        self.conv3 = Sequential([layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu')
                                 ])
        self.conv4 = Sequential([layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
                                 ])
        self.conv5 = Sequential([layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu')
                                 ])
        self.conv6 = Sequential([layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu')
                                 ])
        self.conv7 = Sequential([layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
                                 ])
        self.conv8 = Sequential([layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 ])
        self.conv9 = Sequential([layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu')
                                 ])
        self.conv10 = Sequential([layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
                                  layers.BatchNormalization(),
                                  layers.Activation('relu'),
                                  layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
                                  ])
        self.conv11 = Sequential([layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
                                  layers.BatchNormalization(),
                                  layers.Activation('relu')
                                  ])
        self.conv12 = Sequential([layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
                                  layers.BatchNormalization(),
                                  layers.Activation('relu')
                                  ])
        self.conv13 = Sequential([layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
                                  layers.BatchNormalization(),
                                  layers.Activation('relu'),
                                  layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
                                  ])

        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = Sequential([layers.Dense(2048),
                               layers.BatchNormalization(),
                               layers.Activation('relu'),
                               layers.Dropout(0.5)
                               ])
        self.fc2 = Sequential([layers.Dense(1024),
                               layers.BatchNormalization(),
                               layers.Activation('relu'),
                               layers.Dropout(0.5)
                               ])
        self.fc3 = Sequential([layers.Dense(class_num),
                               layers.BatchNormalization(),
                               layers.Activation('relu'),
                               ])

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)

        return y

    def observe_output(self, inputs, n_im=9, layer=0):
        out_conv_ = []
        x = self.conv1(inputs)
        out_conv_.append(x)
        x = self.conv2(x)
        out_conv_.append(x)
        x = self.conv3(x)
        out_conv_.append(x)
        x = self.conv4(x)
        out_conv_.append(x)
        x = self.conv5(x)
        out_conv_.append(x)
        x = self.conv6(x)
        out_conv_.append(x)
        x = self.conv7(x)
        out_conv_.append(x)
        x = self.conv8(x)
        out_conv_.append(x)
        x = self.conv9(x)
        out_conv_.append(x)
        x = self.conv10(x)
        out_conv_.append(x)
        x = self.conv11(x)
        out_conv_.append(x)
        x = self.conv12(x)
        out_conv_.append(x)
        x = self.conv13(x)
        out_conv_.append(x)

        # print(out_conv_)
        print(out_conv_[0].shape)
        print(len(out_conv_))
        # print((out_conv_[0]))

        # 输出原图
        # img = cv2.resize(inputs[n_im], (256, 256))
        # plt.imshow(img)

        viz_layer(out_conv_, layer=layer, n_filters=64, n_im=n_im)
        viz_layer(out_conv_, layer=layer+1, n_filters=64, n_im=n_im)
        plt.show()
