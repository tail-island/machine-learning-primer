import tensorflow as tf

from dataset import decode, get_test_data_frame, get_xs, get_ys
from translation import translate


# モデルを取得します。
model = tf.keras.models.load_model('model')

# データを取得します。
data_frame = get_test_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# 正解した数。
equal_count = 0

# 実際に予測させて、正解した数を取得します。
for x, y_true, y_pred in zip(xs, ys, translate(model, xs)):
    y_true_string = decode(y_true)
    y_pred_string = decode(y_pred)

    equal = y_true_string == y_pred_string
    equal_count += equal

    print(f'{decode(x)} {"==" if equal else "!="} {y_pred_string}')  # ついでなので、予測結果を出力させます。

# 精度を出力します。
print(f'Accuracy: {equal_count / len(xs)}')
