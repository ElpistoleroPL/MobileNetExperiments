# coding: utf-8

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '89'

np.random.seed(89)
rn.seed(89)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True

from keras import backend as K
tf.set_random_seed(89)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import warnings
import pickle
import os
import os
import datetime
import argparse

from time import time
from scipy import stats
from utils import get_model, get_optimizer

from keras.utils import conv_utils, get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers.core import Lambda
from keras.layers import Input, Activation, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D, MaxPooling2D, Flatten, Dense
from keras.engine.topology import get_source_inputs, InputSpec
from keras.callbacks import LearningRateScheduler
from keras.backend.tensorflow_backend import set_session
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import Nadam, Adam, RMSprop
from keras import layers
from keras import initializers, regularizers, constraints


# Parsing commandline arguments
parser = argparse.ArgumentParser(description='Train and measure activation time')

parser.add_argument("--n", dest="name", type=str, required=True)
parser.add_argument('--b', dest='batch_size', type=int, default=32, help='batch size')
parser.add_argument('--l', dest='lr', type=float, default=None, help='learning rate')
parser.add_argument('--c', dest='config', type=int, default=1, help='configuration')
parser.add_argument('--e', dest='epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--m', dest='model', type=int, default=1, help="Model type")
parser.add_argument('--a', dest='net_param', type=float, default=1.0, help="For MobileNetV1 alpha, for V2 expansion")
parser.add_argument('-s', action="store_true", dest='run_type', default=False, help='if present using smaller dataset')
#parser.add_argument('-t', action="store_true", dest='', default=False, help='if present using smaller dataset')

args = parser.parse_args()

batch_size = args.batch_size
run_type = args.run_type
epochs = args.epochs
model_type = args.model
net_param = args.net_param
name = args.name
configuration = args.config


ALL_EXPERIMENTS_RESULTS = 'results' + name + '.p'

timestamp = int(time())

base_dir = '/home/kgrinholc/ImageNet/CLS-LOC/'
#base_dir_small = '/home/kgrinholc/datasets/places_365_small/'
#base_dir_mac = '/Users/kamil/PG/experiments/data/'
classes = 1000
test_category_batch_size = 10

if run_type:
    base_dir = base_dir_small
    classes = 2

val_dir = base_dir + 'val'
train_dir = base_dir + 'train'
test_dir = base_dir + 'val/'

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    seed=89,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    seed=89,
    class_mode='categorical')

results = {}
class_dict = {v: k for k, v in train_generator.class_indices.items()}
dirs = os.listdir(test_dir)

def scheduler(epoch):
    print(epoch)
    lr = K.get_value(model.optimizer.lr)
    new_lr = lr
    if configuration<7:
        if epoch>1:
            new_lr = lr*0.95
    else:
        if epoch>29 and epoch%30==0:
            new_lr = lr*0.1
    print("Old Lr: " + str(lr) + "; New: " + str(new_lr))
    return new_lr


def compile_and_fit(m, conf):
    optimizer = get_optimizer(conf, lr=args.lr)
    m.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    callbacks = []
    if conf > 2:
        change_lr = LearningRateScheduler(scheduler)
        callbacks.append(change_lr)

    hist = m.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        use_multiprocessing=False,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        callbacks=callbacks,
        shuffle=False
    )
    return hist


def testModel(model, model_name, results, configuration):
    # global i, timeCallback, hist, k, times, d, files, f, img_path, img, x, start, preds, end, handle
    i = model_name
    results[i] = {}
    hist = compile_and_fit(model, configuration)
    for k in hist.history:
        results[i][k] = hist.history[k]
    times = []
    
    with open(ALL_EXPERIMENTS_RESULTS, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('single_results_' + i + '.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if configuration == 0:
    for i in range(1, 5):
        model = get_model(model_type, classes, net_param)
        model.summary()
        testModel(model, name+ '_' + str(i), results, i)
else:
    model = get_model(model_type, classes, net_param)
    model.summary()
    testModel(model, name + '_' + str(configuration), results, configuration)

for i in results.keys():
    print(str(i) + "\t\t" + str(results[i]['val_acc'][epochs - 1]))
