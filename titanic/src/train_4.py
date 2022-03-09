import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat
from sklearn.metrics import accuracy_score


def add_features(data_frame):
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.',)),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',)),
                                  (4, ('Dr.', 'Rev.', 'Don.')),
                                  (5, ('Col.', 'Major.', 'Capt.'))),
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


data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

train_xs = xs[200:]
train_ys = ys[200:]

valid_xs = xs[:200]
valid_ys = ys[:200]

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'force_col_wise': True,
    'feature_pre_filter': False,
    'lambda_l1': 3.7584116683195825,
    'lambda_l2': 1.4648199628787754e-06,
    'num_leaves': 5,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7616580256435892,
    'bagging_freq': 4,
    'min_child_samples': 25
}

cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

for booster in model.boosters:
    print(pd.DataFrame({'feature': booster.feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

plot.plot(cv_result['binary_logloss-mean'])
plot.show()
