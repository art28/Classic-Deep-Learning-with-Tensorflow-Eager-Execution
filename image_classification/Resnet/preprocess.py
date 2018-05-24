import pickle
import numpy as np
from glob import glob

import tensorflow as tf

def load_batches(file_name):
    with open(file_name, 'rb') as f:
        dictionary = pickle.load(f, encoding='latin1')
    return dictionary


def get_X(dictionary):
    return dictionary['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def get_y(dictionary):
    return dictionary['labels']


def prerprocess_train(dir_name):
    X = list()
    y = list()
    for batch_file in glob(dir_name + '*_batch_*'):
        dictionary = load_batches(batch_file)
        X.append(get_X(dictionary))
        y += (get_y(dictionary))

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.array(y)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    return dataset


def prerprocess_test(dir_name):
    X = list()
    y = list()
    for batch_file in glob(dir_name + 'test_batch'):
        dictionary = load_batches(batch_file)
        X.append(get_X(dictionary))
        y += (get_y(dictionary))

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.array(y)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(len(y))
    return dataset
