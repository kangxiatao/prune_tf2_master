# -*- coding: utf-8 -*-

"""
Created on 04/15/2021
exp.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#

"""
- vgg - 
conv2d/kernel:0
conv2d/bias:0
batch_normalization/gamma:0
batch_normalization/beta:0
conv2d_1/kernel:0
conv2d_1/bias:0
batch_normalization_1/gamma:0
batch_normalization_1/beta:0
conv2d_2/kernel:0
conv2d_2/bias:0
batch_normalization_2/gamma:0
batch_normalization_2/beta:0
conv2d_3/kernel:0
conv2d_3/bias:0
batch_normalization_3/gamma:0
batch_normalization_3/beta:0
conv2d_4/kernel:0
conv2d_4/bias:0
batch_normalization_4/gamma:0
batch_normalization_4/beta:0
conv2d_5/kernel:0
conv2d_5/bias:0
batch_normalization_5/gamma:0
batch_normalization_5/beta:0
conv2d_6/kernel:0
conv2d_6/bias:0
batch_normalization_6/gamma:0
batch_normalization_6/beta:0
conv2d_7/kernel:0
conv2d_7/bias:0
batch_normalization_7/gamma:0
batch_normalization_7/beta:0
conv2d_8/kernel:0
conv2d_8/bias:0
batch_normalization_8/gamma:0
batch_normalization_8/beta:0
conv2d_9/kernel:0
conv2d_9/bias:0
batch_normalization_9/gamma:0
batch_normalization_9/beta:0
conv2d_10/kernel:0
conv2d_10/bias:0
batch_normalization_10/gamma:0
batch_normalization_10/beta:0
conv2d_11/kernel:0
conv2d_11/bias:0
batch_normalization_11/gamma:0
batch_normalization_11/beta:0
conv2d_12/kernel:0
conv2d_12/bias:0
batch_normalization_12/gamma:0
batch_normalization_12/beta:0
dense/kernel:0
dense/bias:0
batch_normalization_13/gamma:0
batch_normalization_13/beta:0
dense_1/kernel:0
dense_1/bias:0
batch_normalization_14/gamma:0
batch_normalization_14/beta:0
dense_2/kernel:0
dense_2/bias:0
batch_normalization_15/gamma:0
batch_normalization_15/beta:0
"""

"""
- resnet -
conv2d/kernel:0 --- (3, 3, 3, 64)
conv2d/bias:0 --- (64,)
batch_normalization/gamma:0 --- (64,)
batch_normalization/beta:0 --- (64,)
basic_block/conv2d_1/kernel:0 --- (3, 3, 64, 64)
basic_block/conv2d_1/bias:0 --- (64,)
basic_block/batch_normalization_1/gamma:0 --- (64,)
basic_block/batch_normalization_1/beta:0 --- (64,)
basic_block/conv2d_2/kernel:0 --- (3, 3, 64, 64)
basic_block/conv2d_2/bias:0 --- (64,)
basic_block/batch_normalization_2/gamma:0 --- (64,)
basic_block/batch_normalization_2/beta:0 --- (64,)
basic_block_1/conv2d_3/kernel:0 --- (3, 3, 64, 64)
basic_block_1/conv2d_3/bias:0 --- (64,)
basic_block_1/batch_normalization_3/gamma:0 --- (64,)
basic_block_1/batch_normalization_3/beta:0 --- (64,)
basic_block_1/conv2d_4/kernel:0 --- (3, 3, 64, 64)
basic_block_1/conv2d_4/bias:0 --- (64,)
basic_block_1/batch_normalization_4/gamma:0 --- (64,)
basic_block_1/batch_normalization_4/beta:0 --- (64,)
basic_block_2/conv2d_5/kernel:0 --- (3, 3, 64, 128)
basic_block_2/conv2d_5/bias:0 --- (128,)
basic_block_2/batch_normalization_5/gamma:0 --- (128,)
basic_block_2/batch_normalization_5/beta:0 --- (128,)
basic_block_2/conv2d_6/kernel:0 --- (3, 3, 128, 128)
basic_block_2/conv2d_6/bias:0 --- (128,)
basic_block_2/batch_normalization_6/gamma:0 --- (128,)
basic_block_2/batch_normalization_6/beta:0 --- (128,)
basic_block_2/sequential_3/conv2d_7/kernel:0 --- (1, 1, 64, 128)
basic_block_2/sequential_3/conv2d_7/bias:0 --- (128,)
basic_block_3/conv2d_8/kernel:0 --- (3, 3, 128, 128)
basic_block_3/conv2d_8/bias:0 --- (128,)
basic_block_3/batch_normalization_7/gamma:0 --- (128,)
basic_block_3/batch_normalization_7/beta:0 --- (128,)
basic_block_3/conv2d_9/kernel:0 --- (3, 3, 128, 128)
basic_block_3/conv2d_9/bias:0 --- (128,)
basic_block_3/batch_normalization_8/gamma:0 --- (128,)
basic_block_3/batch_normalization_8/beta:0 --- (128,)
basic_block_4/conv2d_10/kernel:0 --- (3, 3, 128, 256)
basic_block_4/conv2d_10/bias:0 --- (256,)
basic_block_4/batch_normalization_9/gamma:0 --- (256,)
basic_block_4/batch_normalization_9/beta:0 --- (256,)
basic_block_4/conv2d_11/kernel:0 --- (3, 3, 256, 256)
basic_block_4/conv2d_11/bias:0 --- (256,)
basic_block_4/batch_normalization_10/gamma:0 --- (256,)
basic_block_4/batch_normalization_10/beta:0 --- (256,)
basic_block_4/sequential_5/conv2d_12/kernel:0 --- (1, 1, 128, 256)
basic_block_4/sequential_5/conv2d_12/bias:0 --- (256,)
basic_block_5/conv2d_13/kernel:0 --- (3, 3, 256, 256)
basic_block_5/conv2d_13/bias:0 --- (256,)
basic_block_5/batch_normalization_11/gamma:0 --- (256,)
basic_block_5/batch_normalization_11/beta:0 --- (256,)
basic_block_5/conv2d_14/kernel:0 --- (3, 3, 256, 256)
basic_block_5/conv2d_14/bias:0 --- (256,)
basic_block_5/batch_normalization_12/gamma:0 --- (256,)
basic_block_5/batch_normalization_12/beta:0 --- (256,)
basic_block_6/conv2d_15/kernel:0 --- (3, 3, 256, 512)
basic_block_6/conv2d_15/bias:0 --- (512,)
basic_block_6/batch_normalization_13/gamma:0 --- (512,)
basic_block_6/batch_normalization_13/beta:0 --- (512,)
basic_block_6/conv2d_16/kernel:0 --- (3, 3, 512, 512)
basic_block_6/conv2d_16/bias:0 --- (512,)
basic_block_6/batch_normalization_14/gamma:0 --- (512,)
basic_block_6/batch_normalization_14/beta:0 --- (512,)
basic_block_6/sequential_7/conv2d_17/kernel:0 --- (1, 1, 256, 512)
basic_block_6/sequential_7/conv2d_17/bias:0 --- (512,)
basic_block_7/conv2d_18/kernel:0 --- (3, 3, 512, 512)
basic_block_7/conv2d_18/bias:0 --- (512,)
basic_block_7/batch_normalization_15/gamma:0 --- (512,)
basic_block_7/batch_normalization_15/beta:0 --- (512,)
basic_block_7/conv2d_19/kernel:0 --- (3, 3, 512, 512)
basic_block_7/conv2d_19/bias:0 --- (512,)
basic_block_7/batch_normalization_16/gamma:0 --- (512,)
basic_block_7/batch_normalization_16/beta:0 --- (512,)
dense/kernel:0 --- (512, 100)
dense/bias:0 --- (100,)
"""
