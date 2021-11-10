# -*- coding: utf-8 -*-

"""
Created on 03/30/2021
vgg.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
from tensorflow.keras import layers, Sequential

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(tf.keras.Model):
    def __init__(self, vgg_name, class_num=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = layers.Dense(class_num)  # , activation='relu'
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        out = self.features(inputs)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        my_layers = Sequential()
        for x in cfg:
            if x == 'M':
                my_layers.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))
            else:
                my_layers.add(layers.Conv2D(filters=x, kernel_size=(3, 3), padding='same'))
                my_layers.add(layers.BatchNormalization())
                my_layers.add(layers.Activation('relu'))

        my_layers.add(layers.AvgPool2D(pool_size=(1, 1), strides=1))
        return my_layers


def test():
    net = VGG('VGG11')
    net.build(input_shape=(None, 32, 32, 3))
    net.summary()
    # x = tf.random.normal((2, 32, 32, 3), mean=0.0, stddev=1.0)
    # y = net(x)
    # print(y.size())


if __name__ == '__main__':
    test()
