import optuna.integration.lightgbm as lgb
import numpy as np
import pandas as pd

from dataset import get_train_data_frame, get_categorical_features, get_xs, get_ys
from model import save_params


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'regression',
    'metric': 'l2'  # Optunaでエラーが出たので、regressionの場合のデフォルトのmetricを設定しました。
}

# ハイパー・パラメーター・チューニングをします。
tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
cv_result = tuner.run()
model = tuner.get_best_booster()

# 重要な特徴量を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

# LightGBMのパラメーターを保存します。
save_params(tuner.best_params)
