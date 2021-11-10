#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 03/19/2021
loaddata.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split


class Cifar():
    def __init__(self, filename):
        self.filename = filename
        splitfile = filename.split("/")
        metaname = "/batches.meta"
        metalabel = b'label_names'
        if splitfile[-1] == "cifar-100-python":
            metaname = "/meta"
            metalabel = b'fine_label_names'
        self.metaname = metaname
        self.metalabel = metalabel
        self.image_size = 32
        self.img_channels = 3

    # 解析数据
    def unpickle(self, filename):
        with open(filename, "rb") as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # 将数据导入
    def load_data_one(self, file):
        batch = self.unpickle(file)
        data = batch[b'data']
        labelname = b'labels'
        if self.metaname == "/meta":
            labelname = b'fine_labels'
        label = batch[labelname]
        print("Loading %s : %d." % (file, len(data)))
        return data, label

    # 对 label进行处理
    def load_data(self, files, data_dir, label_count):
        data, labels = self.load_data_one(data_dir + "/" + files[0])
        for f in files[1:]:
            data_n, labels_n = self.load_data_one(data_dir + '/' + f)
            data = np.append(data, data_n, axis=0)
            labels = np.append(labels, labels_n, axis=0)
        labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
        data = data.reshape([-1, self.img_channels, self.image_size, self.image_size])
        data = data.transpose([0, 2, 3, 1])  # 将图片转置 跟卷积层一致
        return data, labels

    # 数据导入
    def prepare_data(self):
        print("======Loading data======")
        # data_dir = '../cifar-10-python/cifar-100-python'
        # image_dim = self.image_size * self.image_size * self.img_channels
        meta = self.unpickle(self.filename + self.metaname)
        label_names = meta[self.metalabel]
        label_count = len(label_names)
        if self.metaname == "/batches.meta":
            train_files = ['data_batch_%d' % d for d in range(1, 6)]
            train_data, train_labels = self.load_data(train_files, self.filename, label_count)
            test_data, test_labels = self.load_data(['test_batch'], self.filename, label_count)
        else:
            train_data, train_labels = self.load_data(['train'], self.filename, label_count)
            test_data, test_labels = self.load_data(['test'], self.filename, label_count)
        print("Train data:", np.shape(train_data), np.shape(train_labels))
        print("Test data :", np.shape(test_data), np.shape(test_labels))
        print("======Load finished======")
        print("======Shuffling data======")
        indices = np.random.permutation(len(train_data))  # 打乱数组
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        print("======Prepare finished======")

        return train_data, train_labels, test_data, test_labels


# 对数据随机左右翻转
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


# 随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def data_augmentation(batch, dim=32):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [dim, dim], 4)
    return batch


# Z-score 标准化
def data_standardscaler(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


# Normalization
def data_normalization(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] / 255 - 0.4914) / 0.2023
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] / 255 - 0.4822) / 0.1994
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] / 255 - 0.4465) / 0.2010

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] / 255 - 0.4914) / 0.2023
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] / 255 - 0.4822) / 0.1994
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] / 255 - 0.4465) / 0.2010

    return x_train, x_test


def preprocess(x, y):
    # 数据增强
    padding = 4
    npad = ((padding, padding), (padding, padding), (0, 0))
    x = tf.pad(x, npad)  # padding
    x = tf.image.random_crop(x, [32, 32, 3])  # 随机裁剪
    x = tf.image.random_flip_left_right(x)  # 左右镜像
    # x = tf.image.per_image_standardization(x)  # 图片标准化
    return x, y


def preprocess_test(x, y):
    # x = tf.image.per_image_standardization(x)  # 图片标准化
    return x, y


# def preprocess_mni(x, y):
#     # 数据增强
#     # x = tf.image.random_flip_up_down(x)
#     x = tf.image.random_flip_left_right(x)  # 左右镜像
#     x = tf.image.random_crop(x, [28, 28, 1])  # 随机裁剪
#
#     return x, y


def preprocess_caltech(x, y):
    # 数据增强
    padding = 16
    npad = ((padding, padding), (padding, padding), (0, 0))
    x = tf.pad(x, npad)  # padding
    x = tf.image.random_crop(x, [200, 200, 3])  # 随机裁剪
    x = tf.image.random_flip_left_right(x)  # 左右镜像
    # x = tf.image.per_image_standardization(x)  # 图片标准化
    y = tf.one_hot(y, 102)
    return x, y


def preprocess_caltech_test(x, y):
    y = tf.one_hot(y, 102)
    return x, y


def get_mnist_dataset(args):
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = np.expand_dims(train_images, 3)
    test_images = np.expand_dims(test_images, 3)
    if args.data_name == 'minst2':
        # --- 拆成二分类实验 ---
        train_labels[train_labels > 1] = 1
        test_labels[test_labels > 1] = 1
        train_labels = np.eye(2)[train_labels]
        test_labels = np.eye(2)[test_labels]
    else:
        train_labels = np.eye(10)[train_labels]
        test_labels = np.eye(10)[test_labels]
    print('train_images', train_images.shape)
    print('train_labels', train_labels.shape)
    print('test_images', test_images.shape)
    print('test_labels', test_labels.shape)

    return train_images, train_labels, test_images, test_labels


def ld(args):
    train_images, train_labels, test_images, test_labels = None, None, None, None
    db_train, db_test = None, None
    if 'mnist' in args.data_name:
        train_images, train_labels, test_images, test_labels = get_mnist_dataset(args)
        db_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        db_train = db_train.shuffle(int(train_labels.shape[0] / 5)).batch(args.batch_size)
        db_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        db_test = db_test.batch(args.batch_size)
    elif args.data_name == 'caltech101':
        x = np.load(args.data_dir + '/x%s.npy' % args.image_size,
                    mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        y = np.load(args.data_dir + '/y%s.npy' % args.image_size,
                    mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        train_images, test_images, train_labels, test_labels = train_test_split(x, y, test_size=0.2)
        train_images, test_images = data_standardscaler(train_images, test_images)
        db_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        db_train = db_train.shuffle(int(train_labels.shape[0] / 5)).map(preprocess_caltech).batch(args.batch_size)
        db_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        db_test = db_test.map(preprocess_caltech_test).batch(args.batch_size)
    elif args.data_name == 'cifar10' or args.data_name == 'cifar100':
        # from local
        # raw_data = Cifar(args.data_dir)
        # train_images, train_labels, test_images, test_labels = raw_data.prepare_data()

        # from keras.datasets
        if args.data_name == 'cifar10':
            _dataset = tf.keras.datasets.cifar10
        else:
            _dataset = tf.keras.datasets.cifar100
        (train_images, train_labels), (test_images, test_labels) = _dataset.load_data()
        train_labels = np.eye(args.class_num)[np.squeeze(train_labels)]  # one_hot
        test_labels = np.eye(args.class_num)[np.squeeze(test_labels)]
        print('train_images', train_images.shape)
        print('train_labels', train_labels.shape)
        print('test_images', test_images.shape)
        print('test_labels', test_labels.shape)

        train_images, test_images = data_normalization(train_images, test_images)
        # train_images, test_images = data_standardscaler(train_images, test_images)
        db_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        db_train = db_train.shuffle(int(train_labels.shape[0] / 5)).map(preprocess).batch(args.batch_size)
        db_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        db_test = db_test.map(preprocess_test).batch(args.batch_size)
    # train_set_size = train_labels.shape[0]
    # test_set_size = test_labels.shape[0]

    # return db_train, db_test, train_set_size, test_set_size
    return db_train, db_test, train_images, train_labels, test_images, test_labels
