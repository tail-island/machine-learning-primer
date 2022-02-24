import matplotlib.pyplot as plot
import numpy as np


def check(dataset, model):
    xs, ys = dataset

    plot.plot(xs, ys, 'bo')

    xs_predict = np.linspace(np.min(xs), np.max(xs), 1000).reshape(-1, 1)
    ys_predict = model.predict(xs_predict)

    plot.plot(xs_predict, ys_predict, 'r-')

    plot.show()
