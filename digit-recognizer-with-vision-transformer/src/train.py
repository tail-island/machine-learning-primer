import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from itertools import starmap
from operator import eq
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, DROPOUT_RATE
from vision_transformer import LearningRateSchedule, vision_transformer


rng = np.random.default_rng(0)

data_frame = get_train_data_frame()

xs = get_xs(data_frame)
ys = get_ys(data_frame)

indices = rng.permutation(np.arange(len(xs)))

train_xs = xs[indices[2000:]]
train_ys = ys[indices[2000:]]

valid_xs = xs[indices[:2000]]
valid_ys = ys[indices[:2000]]

op = vision_transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, 28 * 28 // D_MODEL, DROPOUT_RATE)

model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

model.fit(train_xs, train_ys, batch_size=256, epochs=10, validation_data=(valid_xs, valid_ys))

print(f'Accuracy = {sum(starmap(eq, zip(valid_ys, np.argmax(model.predict(valid_xs, batch_size=256), axis=-1)))) / len(valid_xs)}')
