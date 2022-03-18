import lightgbm as lgb
import os.path as path
import pickle

from glob import glob


# LightGBMのパラメーターを保存します。
def save_params(params):
    with open(path.join('digit-recognizer-model', 'params.pickle'), mode='wb') as f:
        pickle.dump(params, f)


# LightGBMのパラメーターを読み込みます。
def load_params():
    with open(path.join('digit-recognizer-model', 'params.pickle'), mode='rb') as f:
        return pickle.load(f)


# モデルを保存します。
def save_model(model):
    for i, booster in enumerate(model.boosters):  # 交差検証なので、複数のモデルが生成されます。
        booster.save_model(path.join('digit-recognizer-model', f'model-{i}.txt'))


# モデルを読み込みます。
def load_model():
    result = lgb.CVBooster()

    for file in sorted(glob(path.join('digit-recognizer-model', 'model-*.txt'))):  # 交差検証なので、複数のモデルが生成されます。
        result.boosters.append(lgb.Booster(model_file=file))

    return result
