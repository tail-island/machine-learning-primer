import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os.path as path

from functools import reduce
from funcy import count, repeat
from glob import glob


# 特長量を追加します。
def add_features(data_frame):
    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    # 家族の人数を追加します。
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    # 料金は合計みたいなので、単価を追加します。
    data_frame['FareUnitPrice'] = data_frame['Fare'] / data_frame['FamilySize']

    return data_frame


# カテゴリ型の特長量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特長量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# Kaggleのnotebookなのかを判定します。
def is_kaggle_notebook():
    return '_dh' in globals() and globals()['_dh'] == ['/kaggle/working']


# モデルをロードします。
def load_model(name):
    result = lgb.CVBooster()

    base_path = '.' if not is_kaggle_notebook() else path.join('..', 'input')  # KaggleのnotebookのDatasetは../inputに展開されます。。。

    for file in sorted(glob(path.join(base_path, 'titanic-model', f'{name}-*.txt'))):  # 交差検証なので、複数のモデルが生成されます。
        result.boosters.append(lgb.Booster(model_file=file))

    return result


# カテゴリ型の特長量を、どの数値に変換するかのdictをロードします。
def load_categorical_features():
    base_path = '.' if not is_kaggle_notebook() else path.join('..', 'input')  # KaggleのnotebookのDatasetは../inputに展開されます。。。

    with open(path.join(base_path, 'titanic-model', 'categorical-features.pickle'), mode='rb') as f:
        return pickle.load(f)


# モデルをロードします。
model = load_model('model')
categorical_features = load_categorical_features()

# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'test.csv')))
data_frame['Fare'] = data_frame['Fare'].fillna(data_frame['Fare'].mean())  # train.csvにはないけど、test.csvのFareにはNaNがある。。。

# 予測用のデータを取得します。
xs = get_xs(data_frame, categorical_features)

# 予測して、結果をCSVとして保存します。
submission = pd.DataFrame({'PassengerId': data_frame['PassengerId'], 'Survived': (np.mean(model.predict(xs), axis=0) >= 0.5).astype(np.int32)})
submission.to_csv('submission.csv', index=False)
