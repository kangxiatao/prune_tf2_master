# -*- coding: utf-8 -*-

"""
Created on 03/19/2021
myhelp.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import os


def get_fit_path(save_dir):

    while os.path.exists(save_dir):
        if save_dir[-1].isdigit():
            path_split = save_dir.split("_")
            for z in range(len(path_split) - 1):
                if z == 0:
                    save_dir = path_split[z]
                else:
                    save_dir += "_" + path_split[z]
            save_dir += "_" + str(int(path_split[-1]) + 1)
        else:
            save_dir = save_dir + "1"

    return save_dir
