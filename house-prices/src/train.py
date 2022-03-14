import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from dataset import get_train_data_frame, get_categorical_features, get_xs, get_ys
from model import load_params
from sklearn.metrics import mean_squared_error


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データセットを取得します。
train_xs = xs[400:]
train_ys = ys[400:]

# 検証データセットを取得します。
valid_xs = xs[:400]
valid_ys = ys[:400]

# LightGBMのパラメーターを取得します。
params = load_params()

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 重要な特徴量を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

# スコアを出力します。
print(f'Score = {np.sqrt(mean_squared_error(np.log(valid_ys), np.log(np.mean(model.predict(valid_xs), axis=0))))}')

# 学習曲線を出力します。
plot.plot(cv_result['l2-mean'])
plot.show()
