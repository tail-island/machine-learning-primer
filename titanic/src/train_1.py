import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from funcy import count
from sklearn.metrics import accuracy_score


def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked')))


def get_xs(data_frame, categorical_features):
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


def get_ys(data_frame):
    return data_frame['Survived']


data_frame = pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv'))
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
    'force_col_wise': True  # 警告を消すために付けました。
}

cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

print(pd.DataFrame({'feature': model.feature_name()[0], 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))
print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

plot.plot(cv_result['binary_logloss-mean'])
plot.show()
