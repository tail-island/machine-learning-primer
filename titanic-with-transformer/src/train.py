import numpy as np
import tensorflow as tf

from dataset import get_categorical_features, get_feature_ranges, get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from itertools import starmap
from operator import eq
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, DROPOUT_RATE
from transformer import LearningRateSchedule, neural_network


rng = np.random.default_rng(0)

data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)
feature_ranges = get_feature_ranges(data_frame)

xs = get_xs(data_frame, categorical_features, feature_ranges)
ys = get_ys(data_frame)

indices = rng.permutation(np.arange(len(xs)))

train_xs = xs[indices[200:]]
train_ys = ys[indices[200:]]

valid_xs = xs[indices[:200]]
valid_ys = ys[indices[:200]]

op = neural_network(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, np.shape(xs)[1], DROPOUT_RATE)

model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='binary_crossentropy', metrics=('accuracy',))
# model.summary()

model.fit(train_xs, train_ys, batch_size=256, epochs=100, validation_data=(valid_xs, valid_ys))

print(f'Accuracy = {sum(starmap(eq, zip(valid_ys, np.argmax(model.predict(valid_xs, batch_size=256), axis=-1)))) / len(valid_xs)}')
