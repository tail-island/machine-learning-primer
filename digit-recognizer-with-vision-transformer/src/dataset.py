import numpy as np
import pandas as pd
import os.path as path

from params import D_MODEL_HEIGHT, D_MODEL_WIDTH


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', 'digit-recognizer', filename))


# 訓練用DataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用DataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 画像をパッチに分割します。
def encode(image):
    def impl():
        for i in range(28 // D_MODEL_HEIGHT):
            for j in range(28 // D_MODEL_WIDTH):
                yield image[i * D_MODEL_HEIGHT: i * D_MODEL_HEIGHT + D_MODEL_HEIGHT, j * D_MODEL_WIDTH: j * D_MODEL_WIDTH + D_MODEL_WIDTH].flatten()

    return np.array(tuple(impl()))


# パッチを画像に戻します。
def decode(encoded):
    result = np.zeros((28, 28), dtype=np.float32)

    for i in range(28 // D_MODEL_HEIGHT):
        for j in range(28 // D_MODEL_WIDTH):
            result[i * D_MODEL_HEIGHT: i * D_MODEL_HEIGHT + D_MODEL_HEIGHT, j * D_MODEL_WIDTH: j * D_MODEL_WIDTH + D_MODEL_WIDTH] = np.reshape(encoded[i * 28 // D_MODEL_WIDTH + j], (D_MODEL_WIDTH, D_MODEL_HEIGHT))

    return result


# 入力データを取得します。
def get_xs(data_frame):
    return np.array(tuple(map(encode, np.reshape(data_frame[list(map(lambda i: f'pixel{i}', range(784)))].values / 255, (-1, 28, 28)))))


# 出力データを取得します。
def get_ys(data_frame):
    return data_frame['label'].values
