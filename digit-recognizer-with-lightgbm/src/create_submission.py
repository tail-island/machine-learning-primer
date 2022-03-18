import numpy as np
import pandas as pd

from dataset import get_test_data_frame, get_xs
from model import load_model


# モデルを読み込みます。
model = load_model()

# データを取得します。
data_frame = get_test_data_frame()
xs = get_xs(data_frame)

# 提出量のCSVを作成します。
submission = pd.DataFrame({'ImageId': data_frame.index + 1, 'Label': np.argmax(np.mean(model.predict(xs), axis=0), axis=-1)})
submission.to_csv('submission.csv', index=False)
