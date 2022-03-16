import numpy as np
import pandas as pd

from funcy import concat, partial, repeatedly
from operator import add, sub


rng = np.random.default_rng(0)


def create_sentence(op_str, op):
    x = rng.integers(1, 1000)
    y = rng.integers(1, 1000)
    z = op(x, y)

    return f'{x}{op_str}{y}', f'{z}'


def create_data_frame(size_per_op):
    expressions, answers = zip(*concat(repeatedly(partial(create_sentence, '+', add), size_per_op),
                                       repeatedly(partial(create_sentence, '-', sub), size_per_op)))

    data_frame = pd.DataFrame({'Expression': expressions, 'Answer': answers}, dtype='string')
    data_frame.index.name = 'Id'

    return data_frame


create_data_frame(10000).to_csv('train.csv')
create_data_frame( 1000).to_csv('test.csv')  # noqa: E201
