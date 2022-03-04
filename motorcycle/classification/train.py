from dataset import get_dataset
from sklearn.ensemble import RandomForestClassifier


# 乱数のシード。
RANDOM_SEED = 0

# データセットを取得します。
train_dataset, valid_dataset, names_collection = get_dataset(RANDOM_SEED)

# モデルを作成して、機械学習します。
model = RandomForestClassifier(random_state=RANDOM_SEED)  # とりあえず、ハイパー・パラメーターはデフォルト値。
model.fit(*train_dataset)

# 訓練データセットを使用して、モデルの精度を表示します。あまり意味はないですけど……。
print(model.score(*train_dataset))

# 検証データセットを使用して、モデルの精度を表示します。
print(model.score(*valid_dataset))

# 検証データセットを使用して……
xs, ys = valid_dataset
_, names = names_collection

# 実際に予測もさせてみます。
for name, y, pred_y, pred_y_proba in zip(names, ys, model.predict(xs), model.predict_proba(xs)):
    print(f'{name}:\t{y}\t{pred_y}\t{pred_y_proba}')
