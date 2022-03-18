import matplotlib.pyplot as plot
import numpy as np
import optuna.integration.lightgbm as lgb

from dataset import get_train_data_frame, get_xs, get_ys
from model import load_params, save_model


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# LightGBMのパラメーターを取得します。
params = load_params()

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 学習曲線を出力します。
plot.plot(cv_result['multi_logloss-mean'])
plot.show()

# モデルを保存します。
save_model(model)
