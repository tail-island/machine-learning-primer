from checker import check
from dataset import get_dataset
from sklearn.linear_model import LinearRegression


# データセットを取得します
dataset = get_dataset()

# モデルを作成して、機械学習します
model = LinearRegression()
model.fit(*dataset)

# 機械学習で作成されたパラメーターを表示します
print(f'アイスクリーム月別支出額 = {model.coef_[0]} * 東京の月平均気温 + {model.intercept_}')

# 図を作成して精度をチェックします
check(dataset, model)
