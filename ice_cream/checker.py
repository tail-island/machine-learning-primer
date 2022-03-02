import matplotlib.pyplot as plot
import numpy as np


def check(dataset, model):
    # 実データを取得します。
    xs, ys = dataset

    # 実データを青色の散布図で描画します。
    plot.plot(xs, ys, 'bo')

    # 予測データを取得します。
    pred_xs = np.linspace(np.min(xs), np.max(xs), 1000).reshape(-1, 1)
    pred_ys = model.predict(pred_xs)

    # 予測データを赤色の線グラフで描画します。
    plot.plot(pred_xs, pred_ys, 'r-')

    # 描画結果を表示します。
    plot.show()
