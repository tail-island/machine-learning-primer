import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from funcy import count
from sklearn.metrics import accuracy_score


# カテゴリ型の特長量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked')))  # factorize()で数値に変換することもできるのですけど、その方式は、実際に予測するときに使えない。。。


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特長量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')  # astype('category')しておけば、LightGBMがカテゴリ型として扱ってくれて便利です。

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# データを読み込んで、前準備をします。
data_frame = pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv'))
categorical_features = get_categorical_features(data_frame)

# 予測用のデータを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データを取得します。
train_xs = xs[200:]
train_ys = ys[200:]

# 検証データを取得します。test.csvを使ってKaggleに問い合わせる方式は、面倒な上に数をこなせないためです。
valid_xs = xs[:200]
valid_ys = ys[:200]

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'binary',  # 2値分類。
    'metric': 'binary_logloss',  # 2値分類の場合は、binary_loglossかauc。
    'force_col_wise': True  # 警告を消すために付けました。
}

# 交差検証で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 各特長量の重要性を出力します。
for booster in model.boosters:
    print(pd.DataFrame({'feature': booster.feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# 精度を出力します。
print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
