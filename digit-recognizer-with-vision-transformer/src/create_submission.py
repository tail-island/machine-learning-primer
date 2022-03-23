import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import get_test_data_frame, get_xs


# モデルを取得します。
model = tf.keras.models.load_model('digit-recognizer-model')

# データを取得します。
data_frame = get_test_data_frame()

# 入力データを取得します。
xs = get_xs(data_frame)

# 解答を作成して、保存します。
submission = pd.DataFrame({'ImageId': data_frame.index + 1, 'Label': np.argmax(model.predict(xs, batch_size=256), axis=-1)})
submission.to_csv('submission.csv', index=False)
