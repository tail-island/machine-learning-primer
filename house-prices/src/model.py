import lightgbm as lgb
import os.path as path
import pickle

from glob import glob


# LightGBMのパラメーターをセーブします。
def save_params(params):
    with open(path.join('house-prices-model', 'params.pickle'), mode='wb') as f:
        pickle.dump(params, f)


# LightGBMのパラメーターをロードします。
def load_params():
    with open(path.join('house-prices-model', 'params.pickle'), mode='rb') as f:
        return pickle.load(f)


# モデルをセーブします。
def save_model(model):
    for i, booster in enumerate(model.boosters):  # 交差検証なので、複数のモデルが生成されます。
        booster.save_model(path.join('house-prices-model', f'model-{i}.txt'))


# モデルをロードします。
def load_model():
    result = lgb.CVBooster()

    for file in sorted(glob(path.join('house-prices-model', 'model-*.txt'))):  # 交差検証なので、複数のモデルが生成されます。
        result.boosters.append(lgb.Booster(model_file=file))

    return result


# カテゴリーの特徴量をセーブします。
def save_categorical_features(categorical_features):
    with open(path.join('house-prices-model', 'categorical-features.pickle'), mode='wb') as f:
        pickle.dump(categorical_features, f)


# カテゴリーの特徴量をロードします。
def load_categorical_features():
    with open(path.join('house-prices-model', 'categorical-features.pickle'), mode='rb') as f:
        return pickle.load(f)
