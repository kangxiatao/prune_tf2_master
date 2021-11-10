# -*- coding: utf-8 -*-

"""
Created on 04/27/2021
caltech101_dataset.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
# from keras import backend as K
from keras.utils import np_utils
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# from modles.googlenet import GoogLeNetBN

# set GPU usage
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
# 导入相应的模块以及进行GPU的设置

# 几个超参数的设计
image_size = 200
classes = 101

root_path = '../../Data/caltech101'
path = root_path + '/101_ObjectCategories'
categories = sorted(os.listdir(path))
ncategories = len(categories)
print(ncategories)
## 设置数据集的路径以及有多少类

def data_process(img_size):
    imgs = []
    labels = []
    img_size = img_size
    size = (img_size, img_size)

    # 限制形状
    # for i, category in enumerate(tqdm(categories)):
    #     for f in os.listdir(path + "/" + categories[i]):
    #         fullpath = os.path.join(path + "/" + categories[i], f)
    #         # print(fullpath)
    #         img = Image.open(fullpath)
    #         img = np.asarray(img.resize(size, Image.ANTIALIAS))
    #         # img = np.asarray(img.resize(size)
    #         if img.shape == (img_size, img_size, 3):
    #             imgs.append(np.array(img))
    #             label_curr = i
    #             labels.append(label_curr)
    #             # imgs_temp = [imgs, labels]
    # np.save(root_path + '/' + 'x'+str(img_size), imgs)
    # np.save(root_path + '/' + 'y'+str(img_size), labels)
# img_size = image_size #设置图片的大小，因为会裁剪图片
# full_path =root_path + '/' + 'x'+str(img_size)
    # 原始形状
    for i, category in enumerate(tqdm(categories)):
        for f in os.listdir(path + "/" + categories[i]):
            fullpath = os.path.join(path + "/" + categories[i], f)
            # print(fullpath)
            img = Image.open(fullpath)
            img = np.asarray(img)
            imgs.append(np.array(img))
            label_curr = i
            labels.append(label_curr)
    np.save(root_path + '/' + 'img_x', imgs)
    np.save(root_path + '/' + 'img_y', labels)
img_size = image_size #设置图片的大小，因为会裁剪图片
full_path =root_path + '/' + 'img_x.npy'
if os.path.exists(full_path) is True:
    print("{} file already exists.".format(full_path))
else:
    data_process(img_size)
## 数据集处理

# x = np.load(root_path + '/x%s.npy' % img_size, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
# y = np.load(root_path + '/y%s.npy' % img_size, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
x = np.load(root_path + '/img_x.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
y = np.load(root_path + '/img_y.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
print("successfully load x%s.npy" % img_size)
## 载入数据 background clutter
plt.imshow(x[100])
plt.show()
## 查看载入是否正确


seed = 7
np.random.seed(seed)
# import pandas as pd
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)
X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)
print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))
# # one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
## 导入数据，拆分为训练集和测试集，0.8:0.2

X_train = X_train.reshape((int(len(X_train)), img_size, img_size, 3))
X_test = X_test.reshape((int(len(X_test)), img_size, img_size, 3))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
## 调整数据的shape


# import numpy as np
# from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
# early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, mode='max')
# csv_logger = CSVLogger('googlenet_caltech101')
# model = GoogLeNetBN(input_shape=(img_size, img_size, 3), classes=classes)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
## 导入模型

#train the model
# Y_train = y_train
# Y_test = y_test
# data_augmentation = False## 是否使用数据增强
# from keras.preprocessing.image import ImageDataGenerator
# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(X_train, Y_train,
#               batch_size=32,
#               nb_epoch=400,
#               validation_data=(X_test, Y_test),
#               shuffle=True,
#               verbose=2,
#               callbacks=[lr_reducer, early_stopper, csv_logger])
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
#
#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(X_train)
#
#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
#                         steps_per_epoch=X_train.shape[0] // 32,
#                         validation_data=(X_test, Y_test),
#                         epochs=400, verbose=2, max_q_size=257,
#                         callbacks=[lr_reducer, early_stopper, csv_logger])


