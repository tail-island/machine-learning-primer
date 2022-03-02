import numpy as np
import os.path as path
import pandas as pd


# 訓練データのインデックスと検証データのインデックスを取得します。
def get_train_indices_and_valid_indices(ys, valid_size, rng):
    size_per_y = min(map(lambda y: len(ys[ys == y]), range(max(ys) + 1)))
    genre_indices_collection = map(lambda y: rng.permutation(rng.choice(np.arange(len(ys))[ys == y], size=size_per_y)), range(max(ys) + 1))
    train_indices_collection, valid_indices_collection = zip(*map(lambda indices: (indices[valid_size:], indices[:valid_size]), genre_indices_collection))

    return np.concatenate(train_indices_collection), np.concatenate(valid_indices_collection)


def get_dataset(seed=0):
    rng = np.random.default_rng(seed)

    # CSVを読み込みます。
    data_frame = pd.read_csv(path.join('..', 'bike-bros-catalog.csv'))

    # 不要なデータを削除します。
    data_frame = data_frame.dropna()                                      # NaN（Not a Number）値がある行を削除します。
    data_frame = data_frame.drop_duplicates(subset=['URL'], keep='last')  # 重複した行を削除します。先の行（メジャーなジャンルの行）は数が多いので、最後の行を残しました。

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
    ys = data_frame['ジャンル'].values
    ns = data_frame['車名'].values

    # 訓練データのインデックスと検証データのインデックスを取得します。
    train_indices, valid_indices = get_train_indices_and_valid_indices(ys, 4, rng)

    # データセットをリターンします。
    return (xs[train_indices], ys[train_indices]), (xs[valid_indices], ys[valid_indices]), (ns[train_indices], ns[valid_indices])


if __name__ == '__main__':
    print(get_dataset(0))
