import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from dense_net import dense_net
from funcy import concat, identity, juxt, partial, repeat, take
from operator import getitem


rng = np.random.default_rng(0)

data_frame = get_train_data_frame()

xs = get_xs(data_frame)
ys = get_ys(data_frame)

op = dense_net(16, 10)

model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=('accuracy',))
model.summary()

batch_size = 128
epoch_size = 40

image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=22.5,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2)

model.fit_generator(image_data_generator.flow(xs, ys, batch_size=batch_size),
                    steps_per_epoch=len(xs) // batch_size,
                    epochs=epoch_size,
                    callbacks=(tf.keras.callbacks.LearningRateScheduler(partial(getitem, tuple(take(epoch_size, concat(repeat(0.01, epoch_size // 2), repeat(0.01 / 10, epoch_size // 4), repeat(0.01 / 100))))))))

model.save('digit-recognizer-model', include_optimizer=False)
