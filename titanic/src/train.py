import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import pickle
import os
import os.path as path

from functools import reduce
from funcy import count, repeat


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


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# 機械学習モデルを保存します。
def save_model(model, name):
    for i, booster in enumerate(model.boosters):  # 交差検証なので、複数のモデルが生成されます。
        booster.save_model(path.join('titanic-model', f'{name}-{i}.txt'))


# カテゴリ型の特長量を、どの数値に変換するかのdictを保存します。
def save_categorical_features(categorical_features):
    with open(path.join('titanic-model', 'categorical-features.pickle'), mode='wb') as f:
        pickle.dump(categorical_features, f)


# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。できるだけ精度を上げたいので、すべてのデータを使用して機械学習します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。Optunaが作成したパラメーター（+ learning_rate）を使用します。
params = {
    'objective': 'binary',
    'force_col_wise': True,
    'feature_pre_filter': False,
    'lambda_l1': 1.4361833756015463,
    'lambda_l2': 2.760985217750726e-07,
    'num_leaves': 5,
    'feature_fraction': 0.4,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'learning_rate': 0.01
}

# 交差検証法で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), num_boost_round=1000, return_cvbooster=True)
model = cv_result['cvbooster']

# モデル保存用のディレクトリを作成します。
os.makedirs('titanic-model', exist_ok=True)

# モデルを保存します。
save_model(model, 'model')
save_categorical_features(categorical_features)

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
