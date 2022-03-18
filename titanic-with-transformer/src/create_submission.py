import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import get_test_data_frame, get_xs
from model import load_categorical_features, load_feature_ranges


# モデルを読み込みます。
model = tf.keras.models.load_model('titanic-model')
categorical_features = load_categorical_features()
feature_ranges = load_feature_ranges()

# データを取得します。
data_frame = get_test_data_frame()
xs = get_xs(data_frame, categorical_features, feature_ranges)

# 提出量のCSVを作成します。
submission = pd.DataFrame({'PassengerId': data_frame['PassengerId'], 'Survived': (model.predict(xs)[:, 0] >= 0.5).astype(np.int32)})
submission.to_csv('submission.csv', index=False)
