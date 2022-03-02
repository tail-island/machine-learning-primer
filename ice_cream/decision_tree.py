from checker import check
from dataset import get_dataset
from sklearn.tree import DecisionTreeRegressor  # , export_text


# データセットを取得します
dataset = get_dataset()

# モデルを作成して、機械学習します
model = DecisionTreeRegressor(max_depth=3)
model.fit(*dataset)

# モデルの内容を可視化します
# print(export_text(model))

# 図を作成して精度をチェックします
check(dataset, model)
