import numpy as np
import tensorflow as tf

from dataset import decode, get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from itertools import starmap
from operator import eq
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, X_VOCAB_SIZE, Y_VOCAB_SIZE, X_MAXIMUM_POSITION, Y_MAXIMUM_POSITION, DROPOUT_RATE
from transformer import LearningRateSchedule, Loss, transformer
from translation import translate


rng = np.random.default_rng(0)

data_frame = get_train_data_frame()

xs = get_xs(data_frame)
ys = get_ys(data_frame)

indices = rng.permutation(np.arange(len(xs)))

train_xs = xs[indices[2000:]]
train_ys = ys[indices[2000:]]

valid_xs = xs[indices[:2000]]
valid_ys = ys[indices[:2000]]

op = transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, X_VOCAB_SIZE, Y_VOCAB_SIZE, X_MAXIMUM_POSITION, Y_MAXIMUM_POSITION, DROPOUT_RATE)

model = tf.keras.Model(*juxt(identity, op)([tf.keras.Input(shape=(None,)), tf.keras.Input(shape=(None,))]))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=Loss(), metrics=())
# model.summary()

model.fit((train_xs, train_ys[:, :-1]), train_ys[:, 1:], batch_size=256, epochs=100, validation_data=((valid_xs, valid_ys[:, :-1]), valid_ys[:, 1:]))
model.save('model', include_optimizer=False)

print(f'Accuracy = {sum(starmap(eq, zip(map(decode, valid_ys), map(decode, translate(model, valid_xs))))) / len(valid_xs)}')
