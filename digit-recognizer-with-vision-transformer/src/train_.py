import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from itertools import starmap
from operator import eq
from params import NUM_BLOCKS, PATCH_HEIGHT, PATCH_WIDTH, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, DROPOUT_RATE
from vision_transformer import LearningRateSchedule, vision_transformer


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# 訓練データセットと検証データセットに分割するためのインデックスを作成します。
indices = rng.permutation(np.arange(len(xs)))

# 訓練データセットを取得します。
train_xs = xs[indices[2000:]]
train_ys = ys[indices[2000:]]

# 検証データセットを取得します。
valid_xs = xs[indices[:2000]]
valid_ys = ys[indices[:2000]]

# Vision Transformerを作成します。
op = vision_transformer(NUM_BLOCKS, PATCH_HEIGHT, PATCH_WIDTH, NUM_HEADS, D_FF, Y_VOCAB_SIZE, 28 * 28 // D_MODEL, DROPOUT_RATE)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

# データの水増しをします。
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=22.5,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2)

batch_size = 256
epoch_size = 150

# 機械学習して、モデルを保存します。
model.fit(image_data_generator.flow(train_xs, train_ys, batch_size=batch_size),
          steps_per_epoch=len(train_xs) // batch_size,
          epochs=epoch_size,
          validation_data=(valid_xs, valid_ys))

# 精度を表示します。
print(f'Accuracy = {sum(starmap(eq, zip(valid_ys, np.argmax(model.predict(valid_xs, batch_size=256), axis=-1)))) / len(valid_xs)}')
