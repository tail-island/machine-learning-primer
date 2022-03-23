import numpy as np
import os.path as path
import pandas as pd

from funcy import concat, count, dropwhile, map, take, takewhile


# 使用される単語。自然言語処理の処理単位は単語です。今回は面倒なので、文字単位にしました。
WORDS = tuple(concat((' ',), ('+', '-'), map(str, range(10)), ('^', '$')))

# 単語を整数にエンコード/デコードするためのdictです。
ENCODES = dict(zip(WORDS, count()))
DECODES = dict(zip(count(), WORDS))


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', filename), dtype={'Expression': 'string', 'Answer': 'string'})


# 訓練用のDataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用のDataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 深層学習するために、文をエンコードして数値の集合に変換します。
def encode(sentence, max_sentence_length):
    return take(max_sentence_length + 2, concat((ENCODES['^'],),  # 文の開始
                                                map(ENCODES, sentence),
                                                (ENCODES['$'],),  # 文の終了
                                                (ENCODES[' '],) * max_sentence_length))  # 残りは空白で埋めます。長さを揃えないと、深層学習できないためです。


# 深層学習が出力した数値の集合を、デコードして文字列に変換します。
def decode(encoded):
    return ''.join(takewhile(lambda c: c != '$', dropwhile(lambda c: c == '^', map(DECODES, encoded))))


# 入力データを取得します。
def get_xs(data_frame):
    strings = data_frame['Expression']
    max_length = max(map(len, strings))

    return np.array(tuple(map(lambda string: tuple(encode(string, max_length)), strings)), dtype=np.int64)


# 正解データを取得します。
def get_ys(data_frame):
    strings = data_frame['Answer']
    max_length = max(map(len, strings))

    return np.array(tuple(map(lambda string: tuple(encode(string, max_length)), strings)), dtype=np.int64)
