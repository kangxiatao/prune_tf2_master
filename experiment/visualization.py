# -*- coding: utf-8 -*-

"""
Created on 04/19/2021
visualization.py.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
# 0.9255    0.9242  0.9225  0.9221  0.9257  0.9207,
# 0.70229	0.71365	0.74083	0.74219	0.7449	0.7495
# 0.9239	0.9173	0.9208	0.9245	0.9229	0.9224
# 0.75013	0.76509	0.74961	0.75407	0.77585	0.76703
# 0.9157	0.9192	0.9196	0.9198	0.9177	0.9167
# 0.80706	0.78873	0.80629	0.81294	0.8125	0.82856
# 0.9119	0.9136	0.9166	0.9086	0.9097	0.9109
# 0.88021	0.86207	0.86176	0.8571	0.85642	0.86551
# 0.9022	0.9031	0.9021	0.9044	0.9069	0.9027
# 0.90936	0.8888	0.8875	0.89775	0.89933	0.88881
# 0.8953	0.897	0.8959	0.8987	0.9002	0.8992
# 0.9233	0.90218	0.90646	0.90798	0.90716	0.89785

"""

# >>> a=array([[10,20],[30,40]])  
# >>> a.repeat([3,2],axis=0)  
# array([[10, 20],  
#        [10, 20],  
#        [10, 20],  
#        [30, 40],  
#        [30, 40]])  
# >>> a.repeat([3,2],axis=1)  
# array([[10, 10, 10, 20, 20],  
#        [30, 30, 30, 40, 40]]) 
gl_x = [0.0008, 0.001, 0.0012, 0.0015, 0.0018, 0.002]
gl_x = np.array(gl_x)
gl_x = np.tile(gl_x.reshape(6, 1), (1, 6))
print(gl_x)
var_y = [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003]
var_y = np.array(var_y)
var_y = np.tile(var_y, (6, 1))
print(var_y)
acc_z = [
    0.9255, 0.9242, 0.9225, 0.9221, 0.9257, 0.9207,
    0.9239, 0.9173, 0.9208, 0.9245, 0.9229, 0.9224,
    0.9157, 0.9192, 0.9196, 0.9198, 0.9177, 0.9167,
    0.9119, 0.9136, 0.9166, 0.9086, 0.9097, 0.9109,
    0.9022, 0.9031, 0.9021, 0.9044, 0.9069, 0.9027,
    0.8953, 0.897, 0.8959, 0.8987, 0.9002, 0.8992
]
prune_z = [
    0.70229, 0.71365, 0.74083, 0.74219, 0.7449, 0.7495,
    0.75013, 0.76509, 0.74961, 0.75407, 0.77585, 0.76703,
    0.80706, 0.78873, 0.80629, 0.81294, 0.8125, 0.82856,
    0.88021, 0.86207, 0.86176, 0.8571, 0.85642, 0.86551,
    0.90936, 0.8888, 0.8875, 0.89775, 0.89933, 0.88881,
    0.9233, 0.90218, 0.90646, 0.90798, 0.90716, 0.89785
]
acc_z = np.array(acc_z).reshape(6, 6)
prune_z = np.array(prune_z).reshape(6, 6)
print(acc_z)
print(prune_z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf1 = ax.plot_surface(gl_x, var_y, acc_z, rstride=1, cstride=1, cmap='autumn')
surf2 = ax.plot_surface(gl_x, var_y, prune_z, rstride=1, cstride=1, cmap='winter')
# ax.plot_wireframe(gl_x, var_y, acc_z, color='c')
# ax.plot_wireframe(gl_x, var_y, prune_z, color='c')
# ax.contour(gl_x, var_y, acc_z, zdir='z', offset=0.7, cmap=plt.get_cmap('rainbow'))
ax.set_xlabel('group lasso', fontsize=12)
ax.set_ylabel('separate angle', fontsize=12)
ax.set_zlabel('acc&prune', fontsize=12)
ax.set_title('accuracy & prune rate', fontsize=20)
# ax.view_init(60, 35)
fig.colorbar(surf2, shrink=0.5, aspect=10)
fig.colorbar(surf1, shrink=0.5, aspect=10)
plt.show()
