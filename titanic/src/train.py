import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import pickle
import os
import os.path as path

from functools import reduce
from funcy import count, repeat


def add_features(data_frame):
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.',)),
                                  (1, ('Mrs.', 'Mme.', 'Ms.')),
                                  (2, ('Miss.',)),
                                  (3, ('Master.', 'Dr.', 'Rev.', 'Don.', 'Major.')),
                                  (4, ('Col.', 'Capt.'))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    return data_frame


def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


def get_xs(data_frame, categorical_features):
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize']]


def get_ys(data_frame):
    return data_frame['Survived']


def save_model(model, name):
    for i, booster in enumerate(model.boosters):
        booster.save_model(path.join('titanic-model', f'{name}-{i}.txt'))


data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# Optunaが作成したパラメーター（+ learning_rate）を使用します。
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'feature_pre_filter': False,
    'lambda_l1': 2.934375313846815,
    'lambda_l2': 4.6970115561476975e-08,
    'num_leaves': 2,
    'feature_fraction': 0.4,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'learning_rage': 0.01
}

cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), num_boost_round=1000, return_cvbooster=True)
model = cv_result['cvbooster']

os.makedirs('titanic-model', exist_ok=True)

save_model(model, 'model')

with open(path.join('titanic-model', 'categorical-features.pickle'), mode='wb') as f:
    pickle.dump(categorical_features, f)

plot.plot(cv_result['binary_logloss-mean'])
plot.show()
