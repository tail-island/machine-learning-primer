import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat
from sklearn.metrics import accuracy_score


# 特徴量を追加します。
def add_features(data_frame):
    # train.csvでの肩書の内訳は、以下の通り。
    # Mr.        509
    # Miss.      180
    # Mrs.       125
    # Master.     40  少年もしくは青年への敬称らしい
    # Dr.         11
    # Col.        10
    # Rev.         6  聖職者への敬称らしい
    # Don.         2
    # Major.       2
    # Mme.         1
    # Ms.          1
    # Capt.        1
    # NaN.         3

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

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データセットを取得します。
train_xs = xs[200:]
train_ys = ys[200:]

# 検証データセットを取得します。test.csvを使ってKaggleに問い合わせる方式は、面倒な上に数をこなせないためです。
valid_xs = xs[:200]
valid_ys = ys[:200]

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'binary',  # 2値分類。
    'force_col_wise': True  # 警告を消すために付けました。
}

# 交差検証で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 特徴量の重要性を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# 精度を出力します。
print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
