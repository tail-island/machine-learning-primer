import optuna.integration.lightgbm as lgb

from dataset import get_train_data_frame, get_xs, get_ys
from model import save_params


# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 10,
    'force_col_wise': True  # LightGBMの警告を除去するために追加しました。
}

# ハイパー・パラメーター・チューニングをします。
tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
cv_result = tuner.run()
model = tuner.get_best_booster()

# LightGBMのパラメーターを保存します。
save_params(tuner.best_params)
