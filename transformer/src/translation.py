import numpy as np

from dataset import ENCODES
from params import Y_MAXIMUM_POSITION


def translate(model, xs):
    ys = np.zeros((len(xs), Y_MAXIMUM_POSITION), dtype=np.int64)
    ys[:, 0] = ENCODES['^']

    for i in range(1, Y_MAXIMUM_POSITION):
        ys[:, i] = np.argmax(model.predict((xs, ys[:, :i]), batch_size=256)[:, -1], axis=-1)

        if np.all(ys[:, i] == ENCODES['$']):
            break

    return ys
