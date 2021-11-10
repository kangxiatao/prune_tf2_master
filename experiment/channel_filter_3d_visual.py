# -*- coding: utf-8 -*-

"""
Created on 04/10/2021
channel_filter_3d_visual.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_2d_data(data_file):
    _filter = []
    _channel = []
    _data2d = []
    f = h5py.File(data_file, "r")
    for root_name, g in f.items():
        # print(root_name)
        for _, weights_dirs in g.attrs.items():
            for i in weights_dirs:
                name = root_name + "/" + str(i, encoding="utf-8")
                data = f[name]
                # print(data.name)
                # print(data.value)
                if 'conv' in data.name and 'kernel' in data.name:
                    # 读取每层数据
                    layer_data = data.value
                    _filter.append(np.sum(np.abs(layer_data), axis=(0, 1, 2)))
                    _data2d.append(layer_data)
    return _filter, _data2d


# gl_15 = '../trained_model/test/gl_15.h5'
# var_15 = '../trained_model/test/var_15.h5'
# gl_08 = '../trained_model/test/gl_08.h5'
# var_25 = '../trained_model/test/var_25.h5'
gl_12 = '../trained_model/test/gl_12.h5'
var_05 = '../trained_model/test/var_05.h5'
# prune_filter, prune_data2d = load_2d_data(gl_12)
prune_filter, prune_data2d = load_2d_data(var_05)
# print(prune_data2d)
# print(len(prune_data2d))
# print(len(prune_data2d[0]))
# print(len(prune_data2d[0][0]))
# print(len(prune_data2d[0][0][0]))
# print(len(prune_data2d[0][0][0][0]))

"""
惩罚依次为：
    无 - 精度：91.27%，剪枝率：- （坐标轴限制：0.3）
    L2 - 精度：93.50%，剪枝率：- （坐标轴限制：0.05）
    gl - 精度：91.07%，剪枝率：55.76% （坐标轴限制：0.02）
    相似角 - 精度：92.01%，剪枝率：52.16%（坐标轴限制：0.02）
"""


num_look = 36
lim_v = 0.2
# for layer in range(13):
layer = 6
if 1:
    # 取当前层
    aaa = prune_data2d[layer]  # 全部
    # [k, k, c, n] => [c, n, k, k]
    bbb = np.swapaxes(np.swapaxes(aaa, 0, 2), 1, 3)
    b_shape = bbb.shape
    # 把过滤器合并成向量
    # [c, n, k, k] => [c, n, k*k]
    # ccc = bbb.reshape(b_shape[0], b_shape[1], b_shape[2]*b_shape[3])
    # [c, n, k, k] => [c, n, k]
    ccc = np.mean(bbb, 3)
    # [c, n, k] => [c, k, n]
    ddd = np.swapaxes(ccc, 1, 2)

    fig = plt.figure(figsize=(20, 20))
    cnt = 1
    for i in range(0, b_shape[0], int(b_shape[0]/num_look)):

        ax = fig.add_subplot(int(math.sqrt(num_look)), int(math.sqrt(num_look)), cnt, projection='3d')
        ax.scatter(ddd[i][0], ddd[i][1], ddd[i][2], s=45)
        ax.set_xlim(-lim_v, lim_v)
        ax.set_ylim(-lim_v, lim_v)
        ax.set_zlim(-lim_v, lim_v)
        ax.set_title('channel %s' % str(i + 1), fontsize=30)
        cnt += 1
        if cnt > num_look:
            break
    plt.show()

    # # [c, k, n] => [n, k, c]
    # ddd = np.swapaxes(ddd, 0, 2)
    #
    # fig = plt.figure(figsize=(20, 20))
    # cnt = 1
    # for i in range(0, b_shape[1], int(b_shape[1]/num_look)):
    #
    #     ax = fig.add_subplot(int(math.sqrt(num_look)), int(math.sqrt(num_look)), cnt, projection='3d')
    #     ax.scatter(ddd[i][0], ddd[i][1], ddd[i][2], s=45)
    #     ax.set_xlim(-lim_v, lim_v)
    #     ax.set_ylim(-lim_v, lim_v)
    #     ax.set_zlim(-lim_v, lim_v)
    #     ax.set_title('filter %s' % str(i + 1), fontsize=30)
    #     cnt += 1
    #     if cnt > num_look:
    #         break
    # plt.show()

