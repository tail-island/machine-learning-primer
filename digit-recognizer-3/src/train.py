import matplotlib.pyplot as plot
import numpy as np
import optuna.integration.lightgbm as lgb

from dataset import get_train_data_frame, get_xs, get_ys
from model import load_params
from sklearn.metrics import accuracy_score


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# データセット分割用のインデックスを作成します。
indices = rng.permutation(np.arange(len(xs)))

# 訓練データセットを取得します。
train_xs = xs.iloc[list(indices[2000:])]
train_ys = ys[indices[2000:]]

# 検証データセットを取得します。
valid_xs = xs.iloc[indices[:2000]]
valid_ys = ys[indices[:2000]]

# LightGBMのパラメーターを取得します。
params = load_params()

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 精度を出力します。
print(f'Accuracy = {accuracy_score(valid_ys, np.argmax(np.mean(model.predict(valid_xs), axis=0), axis=-1))}')

# 学習曲線を出力します。
plot.plot(cv_result['multi_logloss-mean'])
plot.show()
