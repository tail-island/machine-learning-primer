import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from dense_net import dense_net
from funcy import concat, identity, juxt, partial, repeat, take
from itertools import starmap
from operator import eq, getitem


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

# DenseNetを作成します。
op = dense_net(16, 10)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

batch_size = 128
epoch_size = 40

# データの水増しをします。
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=22.5,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2)

# 機械学習して、モデルを保存します。
model.fit_generator(image_data_generator.flow(train_xs, train_ys, batch_size=batch_size),
                    steps_per_epoch=len(train_xs) // batch_size,
                    epochs=epoch_size,
                    callbacks=(tf.keras.callbacks.LearningRateScheduler(partial(getitem, tuple(take(epoch_size, concat(repeat(0.01, epoch_size // 2), repeat(0.01 / 10, epoch_size // 4), repeat(0.01 / 100))))))),
                    validation_data=(valid_xs, valid_ys))

# 精度を出力します。
print(f'Accuracy = {sum(starmap(eq, zip(valid_ys, np.argmax(model.predict(valid_xs, batch_size=256), axis=-1)))) / len(valid_xs)}')
