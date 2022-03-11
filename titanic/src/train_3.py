import optuna.integration.lightgbm as lgb
import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat


# 特徴量を追加します。
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


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。Optunaを信用しているので検証は不要と考え、検証データは作成しません。データ量が多い方が正確なハイパー・パラメーターになりますし。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'binary',  # 2値分類。
    'force_col_wise': True  # 警告を消すために付けました。
}

# 交差検証でハイパー・パラメーター・チューニングをします。
tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
cv_result = tuner.run()
model = tuner.get_best_booster()

# 特徴量の重要性を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# ハイパー・パラメーターを出力します。
print(tuner.best_params)
