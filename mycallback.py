# -*- coding: utf-8 -*-

"""
Created on 04/13/2021
callback.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import tensorflow as tf
import numpy as np
from tensorflow import keras


class ModelSaveToH5:
    def __init__(self,
                 model,
                 filepath,
                 monitor='val_loss',
                 mode='auto',
                 verbose=0):
        self.model = model
        self._supports_tf_logs = True
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # print(logs)
        self._save_model(epoch=epoch, logs=logs)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self.filepath

        current = logs.get(self.monitor)

        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
            self.best = current
            self.model.save_weights(filepath)
        else:
            if self.verbose > 0:
                print('Epoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))


class EarlyStopping:
    def __init__(self,
                 train_model,
                 monitor='val_loss',
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None):

        self.train_model = train_model
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current, self.best):
            # patience=1，Stop at a certain accuracy
            if self.patience == 1:
                self.stopped_epoch = epoch
                self.train_model.stop_training = True
            else:
                self.best = current
                self.wait = 0
        else:
            if self.patience == 1:
                pass
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.train_model.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        return monitor_value
