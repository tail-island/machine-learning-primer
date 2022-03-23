import numpy as np
import tensorflow as tf

from dataset import get_feature_ranges, get_train_data_frame, get_categorical_features, get_xs, get_ys
from funcy import identity, juxt
from model import save_feature_rangess, save_categorical_features
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, DROPOUT_RATE
from transformer import LearningRateSchedule, neural_network


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)
feature_ranges = get_feature_ranges(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features, feature_ranges)
ys = get_ys(data_frame)

# 機械学習します。
op = neural_network(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, np.shape(xs)[1], DROPOUT_RATE)
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='binary_crossentropy', metrics=('accuracy',))
model.fit(xs, ys, batch_size=256, epochs=100)

# モデルを保存します。
model.save('titanic-model', include_optimizer=False)
save_categorical_features(categorical_features)
save_feature_rangess(feature_ranges)
