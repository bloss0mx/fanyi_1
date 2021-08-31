# from train_ctr import train
import jieba
import os
import random
import re
from numpy.lib.twodim_base import tri
from tensorflow.python.keras.layers.core import Lambda
from tqdm import tqdm
######################################################
# encoding: utf-8
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers
# from tensorflow import keras
import os
import pymysql
import time
import datetime
import pickle
import json
import sys
import numpy as np
from multiprocessing import Process, Queue

from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor

import tensorflow_hub as hub
import tensorflow_text
######################################################
from tools import loadRcdsFromFile
from itertools import chain
import itertools
from collections import Counter


modal_path = './3rd-party-model/universal-sentence-encoder-multilingual-large_3'


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def withNoneSplit(train_data):
    cutted = train_data
    return cutted


def train_ebd(train_file_path):
    train_data = loadRcdsFromFile(train_file_path)
    train_data = list(map(lambda x: x[0], train_data))
    train_data = withNoneSplit(train_data)

    with open('./train_text.txt', 'w') as f:
        for line in train_data:
            f.write(line)
            f.write('\n')

    origin_data = train_data

    train_data = list(chunks(train_data, 500))
    embed = hub.load(modal_path)
    ebd = list(map(lambda x: embed(x), tqdm(train_data)))
    ebd = list(chain.from_iterable(ebd))

    testIdx = 50001
    innered = np.inner(ebd[testIdx], ebd)
    answer = np.argsort(-np.array(innered))
    print('\na:', origin_data[testIdx])
    for i in answer[:20]:
        print(i, innered[i], origin_data[i])

    return ebd
