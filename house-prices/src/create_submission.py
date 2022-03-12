import numpy as np
import pandas as pd

from dataset import get_test_data_frame, get_xs
from model import load_categorical_features, load_model


model = load_model()
categorical_features = load_categorical_features()

data_frame = get_test_data_frame()
xs = get_xs(data_frame, categorical_features)

submission = pd.DataFrame({'Id': data_frame['Id'], 'SalePrice': np.mean(model.predict(xs), axis=0)})
submission.to_csv('submission.csv', index=False)
