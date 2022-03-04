import numpy as np
import os.path as path
import pandas as pd


def get_dataset():
    # CSVを読み込みます。
    data_frame = pd.read_csv(path.join('..', 'bike-bros-catalog.csv'))

    # 不要なデータを削除します。
    data_frame = data_frame.dropna()                         # NaN（Not a Number）値がある行を削除します。
    data_frame = data_frame.drop_duplicates(subset=['URL'])  # 重複した行を削除します。

    # 列を選択します。
    xs = pd.get_dummies(data_frame[['価格',
                                    '全長 (mm)',
                                    '全幅 (mm)',
                                    '全高 (mm)',
                                    'ホイールベース (mm)',
                                    'シート高 (mm)',
                                    '車両重量 (kg)',
                                    '気筒数',
                                    'シリンダ配列',
                                    '排気量 (cc)',
                                    'カム・バルブ駆動方式',
                                    '気筒あたりバルブ数',
                                    '最高出力（kW）',
                                    '最高出力回転数（rpm）',
                                    '最大トルク（N・m）',
                                    '最大トルク回転数（rpm）']],
                        columns=['シリンダ配列', 'カム・バルブ駆動方式']).values
    ns = data_frame['車名'].values

    # 0～1の値に正規化します。
    xs_max = np.max(xs, axis=0)
    xs_min = np.min(xs, axis=0)
    xs = (xs - xs_min) / (xs_max - xs_min)

    # データセットをリターンします。
    return xs, ns


if __name__ == '__main__':
    xs, ns = get_dataset(0)

    for x, n in zip(xs, ns):
        print(n, x)
