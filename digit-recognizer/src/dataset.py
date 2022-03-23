import numpy as np
import pandas as pd
import os.path as path


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', 'digit-recognizer', filename))


# 訓練用DataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用DataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 入力データを取得します。
def get_xs(data_frame):
    return np.array(np.reshape(data_frame[list(map(lambda i: f'pixel{i}', range(784)))].values / 255, (-1, 28, 28, 1)))


# 出力データを取得します。
def get_ys(data_frame):
    return data_frame['label'].values
