import pandas as pd
import os.path as path


def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', 'digit-recognizer', filename), dtype={'Expression': 'string', 'Answer': 'string'})


def get_train_data_frame():
    return get_data_frame('train.csv')


def get_test_data_frame():
    return get_data_frame('test.csv')


def get_xs(data_frame):
    return data_frame[list(map(lambda i: f'pixel{i}', range(784)))]


def get_ys(data_frame):
    return data_frame['label'].values
