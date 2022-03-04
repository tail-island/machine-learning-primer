from dataset import get_dataset
from sklearn.cluster import KMeans


# 乱数のシード。
RANDOM_SEED = 0

# クラスタの数
CLUSTER_SIZE = 4

# データセットを取得します。
xs, names = get_dataset()

# モデルを作成して、機械学習します。
model = KMeans(n_clusters=CLUSTER_SIZE, random_state=RANDOM_SEED)
ys = model.fit_predict(xs)

# クラスタを表示させてみます。
for i in range(CLUSTER_SIZE):
    for name in sorted(names[ys == i]):
        print(f'{name}')
    print()
