import numpy as np
import tensorflow as tf

from dataset import ENCODES, decode, get_test_data_frame, get_xs, get_ys
from params import Y_MAXIMUM_POSITION


def translate(model, xs):
    ys = np.zeros((len(xs), Y_MAXIMUM_POSITION), dtype=np.int64)
    ys[:, 0] = ENCODES['^']

    for i in range(1, Y_MAXIMUM_POSITION):
        ys[:, i] = np.argmax(model.predict((xs, ys[:, :i]))[:, -1], axis=-1)

        if np.all(ys[:, i] == ENCODES['$']):
            break

    return ys


model = tf.keras.models.load_model('model')

data_frame = get_test_data_frame()

xs = get_xs(data_frame)
ys = get_ys(data_frame)

equal_count = 0

for x, y_true, y_pred in zip(xs, ys, translate(model, xs)):
    y_true_string = decode(y_true)
    y_pred_string = decode(y_pred)

    equal = y_true_string == y_pred_string
    equal_count += equal

    print(f'{decode(x)} {"==" if equal else "!="} {y_pred_string}')

print(f'Accuracy: {equal_count / len(xs)}')
