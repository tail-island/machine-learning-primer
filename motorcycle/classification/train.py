from dataset import get_dataset
from sklearn.ensemble import RandomForestClassifier


# データセットを取得します。
train_dataset, valid_dataset, names_collection = get_dataset()

# モデルを作成して、機械学習します。
model = RandomForestClassifier(random_state=0)  # とりあえず、ハイパー・パラメーターはデフォルト値。
model.fit(*train_dataset)

# 訓練データセットを使用して、モデルの精度を表示します。
print(model.score(*train_dataset))

# 検証データセットを使用して、モデルの精度を表示します。
print(model.score(*valid_dataset))

# 検証データセットを使用して……
xs, ys = valid_dataset
_, names = names_collection

# 実際に予測させてみます。
for name, y, pred_y in zip(names, ys, model.predict_proba(xs)):
    print(f'{name}:\t{y}\t{pred_y}')
