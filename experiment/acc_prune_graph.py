# -*- coding: utf-8 -*-

"""
Created on 05/12/2021
acc_prune_graph.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = [0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002]
y_acc = [0.9255, 0.9239, 0.9157, 0.9119, 0.9022, 0.8953]
y_prune = [0.70229, 0.75013, 0.80706, 0.88021, 0.90936, 0.9233]

plt.figure(figsize=(6, 4))
plt.plot(x, y_acc, color="red", linewidth=1)
plt.plot(x, y_prune, color="blue", linewidth=1)
plt.xlabel("Punishment")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
plt.ylabel("accuracy & prune rate")
plt.title("Group Lasso")  # title：设置子图的标题。
# plt.ylim(-1.1, 1.1)  # xlim、ylim：分别设置X、Y轴的显示范围。
plt.show()
# plt.close()

