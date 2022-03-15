import numpy as np
import os.path as path
import pandas as pd

from funcy import concat, count, dropwhile, map, take, takewhile


WORDS = tuple(concat((' ',), ('+', '-'), map(str, range(10)), ('^', '$')))

ENCODES = dict(zip(WORDS, count()))
DECODES = dict(zip(count(), WORDS))


def get_train_data_frame():
    return pd.read_csv(path.join('..', 'input', 'train.csv'),
                       dtype={'Expression': 'string', 'Answer': 'string'})


def get_test_data_frame():
    return pd.read_csv(path.join('..', 'input', 'test.csv'),
                       dtype={'Expression': 'string', 'Answer': 'string'})


def encode(sentence, max_sentence_length):
    return take(max_sentence_length + 2, concat((ENCODES['^'],),
                                                map(ENCODES, sentence),
                                                (ENCODES['$'],),
                                                (ENCODES[' '],) * max_sentence_length))


def decode(encoded):
    return ''.join(takewhile(lambda c: c != '$', dropwhile(lambda c: c == '^', map(DECODES, encoded))))


def get_xs(data_frame):
    strings = data_frame['Expression']
    max_length = max(map(len, strings))

    return np.array(tuple(map(lambda string: tuple(encode(string, max_length)), strings)), dtype=np.int64)


def get_ys(data_frame):
    strings = data_frame['Answer']
    max_length = max(map(len, strings))

    return np.array(tuple(map(lambda string: tuple(encode(string, max_length)), strings)), dtype=np.int64)
