import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from dataset import get_train_data_frame, get_categorical_features, get_xs, get_ys
from model import load_params, save_model, save_categorical_features


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを取得します。
params = load_params() | {'learning_rate': 0.01}

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, num_boost_round=2000)
model = cv_result['cvbooster']

# 重要な特徴量を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

# 学習曲線を出力します。
plot.plot(cv_result['l2-mean'])
plot.show()

# モデルを保存します。
save_model(model)
save_categorical_features(categorical_features)
