import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os.path as path

from functools import reduce
from funcy import count, repeat
from glob import glob


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
        data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')

    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize']]


def load_model(name):
    result = lgb.CVBooster()

    for file in sorted(glob(f'{name}-*.txt')):
        result.boosters.append(lgb.Booster(model_file=file))

    return result


data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'test.csv')))
with open('categorical_features.pickle', mode='rb') as f:
    categorical_features = pickle.load(f)

xs = get_xs(data_frame, categorical_features)
model = load_model('model')

answer = pd.DataFrame({'PassengerId': data_frame['PassengerId'], 'Survived': (np.mean(model.predict(xs), axis=0) >= 0.5).astype(np.int32)})
answer.to_csv('answer.csv', index=False)
