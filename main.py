# -*- coding: utf-8 -*-

"""
Created on 03/18/2021
main.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import numpy as np
import math
import tensorflow as tf
import models.lenet5 as lenet5
import models.vgg16 as vgg16
import models.resnet as resnet
import utility.loaddata as ld
from utility.log_helper import *
from utility.cosine_lr import *
import myparser
import penalty
import prune
import train


# --- set tf ---
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# --- set args ---
args = myparser.parse_args()
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# --- create log ---
logging_config(folder=args.save_dir, name='running', no_console=False)

# --- tensorboard logdir ---
# log_dir = args.save_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# print("tensorboard --logdir {}".format(args.save_dir))

# --- Load  dataset ---
db_train, db_test, train_images, train_labels, test_images, test_labels = ld.ld(args)
args.train_set_size = train_labels.shape[0]
args.test_set_size = test_labels.shape[0]

# --- Define the model architecture ---
model = None
if args.model == 'lenet':
    model = lenet5.LeNet5(args.class_num)
elif args.model == 'vgg':
    model = vgg16.VGG16(args.class_num)
    # model = vgg.VGG('VGG16', args.class_num)
elif args.model == 'resnet':
    model = resnet.resnet18(args.class_num)
elif args.model == 'resnet34':
    model = resnet.resnet34(args.class_num)
else:
    model = lenet5.LeNet5(args.class_num)
model.build(input_shape=(None, args.image_size, args.image_size, args.image_channel))

# --- optimizer ---
# optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr)
optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr, momentum=0.9)

# --- learning rate ---
cnt_step = math.ceil(args.train_set_size / args.batch_size)
total_steps = int(args.epochs * args.train_set_size / args.batch_size)
# Create the Learning rate scheduler.
cos_lr = WarmUpCosineDecayScheduler(learning_rate_base=args.init_lr,
                                    total_steps=total_steps,
                                    warmup_learning_rate=0.001,
                                    warmup_steps=10,
                                    # verbose=1,
                                    # log_dir=log_dir,
                                    optimizer=optimizer,
                                    )

# --- compile  model ---
train_model = train.Train(model, args, db_train, db_test, optimizer, cos_lr)
# for i, weight in enumerate(model.trainable_variables):
#     print(weight.name, '---', weight.get_shape())
    # if 'conv' in weight.name and 'kernel' in weight.name:  # conv2d
    #     print(weight.name, '---', weight.get_shape())

# --- reverse ---
if args.is_restore:
    model.load_weights(args.restore_path + 'anoi.h5')

# --- prior pruning ---
args.prior_prune_bool_list = None
if args.prior_prune and args.is_restore:
    _accuracy, _prune_rate, args.prior_prune_bool_list = prune.prior_pruning(model, db_test, args, 'auto')
    logging.info(
        '--- prior pruning --- threshold: auto => accuracy: {:.5f} | prune rate: {:.5f}'.format(_accuracy, _prune_rate))

# --- Train ---
if args.train:
    train_model()
    model.load_weights(args.save_dir + 'anoi.h5')  # get best weights
    baseline_model_accuracy = train_model.evaluate(db_test, verbose=0)[1]
    logging.info('Baseline test accuracy: {:.5f}'.format(baseline_model_accuracy))

# --- save model to h5 ---
if args.store_weight:
    keras_file = args.baseline_keras + '.h5'
    model.save_weights(keras_file)
    logging.info('Saved baseline model to: {}'.format(keras_file))

# --- Evaluate  model ---
if args.test:
    baseline_model_accuracy = train_model.evaluate(db_test, verbose=0)[1]
    logging.info('Baseline test accuracy: {:.5f}'.format(baseline_model_accuracy))

# --- Pruning ---
if args.prune:
    _accuracy, _prune_rate = prune.start_pruning(model, train_model.evaluate, db_test, args, 'auto')
    logging.info('threshold: auto => accuracy: {:.5f} | prune rate: {:.5f}'.format(_accuracy, _prune_rate))
    # model.load_weights(args.save_dir + 'anoi.h5')  # get best weights
    # _accuracy, _prune_rate = prune.start_pruning(model, db_test, args)
    # logging.info('threshold: {} => accuracy: {:.5f} | prune rate: {:.5f}'.format(args.threshold, _accuracy, _prune_rate))

# --- Observe output ---
# if args.model == 'vgg':
#     model.observe_output(test_images[:32], 9, 0)
