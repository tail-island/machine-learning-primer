import os.path as path
import pickle


# カテゴリーの特徴量を保存します。
def save_categorical_features(categorical_features):
    with open(path.join('titanic-model', 'categorical-features.pickle'), mode='wb') as f:
        pickle.dump(categorical_features, f)


# カテゴリーの特徴量を読み込みます。
def load_categorical_features():
    with open(path.join('titanic-model', 'categorical-features.pickle'), mode='rb') as f:
        return pickle.load(f)


# 数値型の特徴量の最小値と最大値を保存します。
def save_feature_rangess(feature_ranges):
    with open(path.join('titanic-model', 'feature-ranges.pickle'), mode='wb') as f:
        pickle.dump(feature_ranges, f)


# 数値型の特徴量の最小値と最大値を読み込みます。
def load_feature_ranges():
    with open(path.join('titanic-model', 'feature-ranges.pickle'), mode='rb') as f:
        return pickle.load(f)
