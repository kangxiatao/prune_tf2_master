# -*- coding: utf-8 -*-

"""
Created on 03/18/2021
pruning.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

#
import numpy as np
import math
import tensorflow as tf
import models.lenet5 as lenet5
import models.vgg as vgg
import models.vgg16 as vgg16
import models.resnet as resnet
import utility.loaddata as ld
from utility.log_helper import *
from utility.cosine_lr import *
import prune
import myparser
import train_tpu

# --- set args ---
args = myparser.parse_args()
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# --- use tpu ---
print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    raise BaseException(
        'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('tpu_strategy.num_replicas_in_sync:', tpu_strategy.num_replicas_in_sync)
args.batch_size = args.batch_size // tpu_strategy.num_replicas_in_sync
args.strategy_num = tpu_strategy.num_replicas_in_sync

# --- create log ---
logging_config(folder=args.save_dir, name='running', no_console=False)
# log.logging.info(args)

# --- Load  dataset ---
train_set, test_set, train_images, train_labels, test_images, test_labels = ld.ld(args)
args.train_set_size = train_labels.shape[0]
args.test_set_size = test_labels.shape[0]
# dataset to gpu
db_train = tpu_strategy.experimental_distribute_datasets_from_function(lambda _: train_set)
db_test = tpu_strategy.experimental_distribute_datasets_from_function(lambda _: test_set)


# --- creating the model in TPU ---
with tpu_strategy.scope():  # creating the model in the TPUStrategy scope means we will train the model on the TPU
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
    train_model = train_tpu.Train(model, tpu_strategy, args, db_train, db_test, optimizer, cos_lr)

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
    # _accuracy, _prune_rate = prune.start_pruning(model, train_model.evaluate, db_test, args)
    # logging.info('threshold: {} => accuracy: {:.5f} | prune rate: {:.5f}'.format(args.threshold, _accuracy, _prune_rate))
