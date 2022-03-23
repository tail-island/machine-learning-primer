import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, DROPOUT_RATE
from vision_transformer import LearningRateSchedule, vision_transformer


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# Vision Transformerを作成します。
op = vision_transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, 28 * 28 // D_MODEL, DROPOUT_RATE)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

# 機械学習して、モデルを保存します。
model.fit(xs, ys, batch_size=256, epochs=100)
model.save('digit-recognizer-model', include_optimizer=False)
