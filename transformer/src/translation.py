import numpy as np

from dataset import ENCODES
from params import Y_MAXIMUM_POSITION


# 翻訳します。
def translate(model, xs):
    # 仮の翻訳結果を作成し、文の開始記号を設定します。
    ys = np.zeros((len(xs), Y_MAXIMUM_POSITION), dtype=np.int64)
    ys[:, 0] = ENCODES['^']

    # 文の終了記号が出力されたかを表現する変数です。
    is_ends = np.zeros((len(xs),), dtype=np.int32)

    # Transformerは、学習時は文の単位なのですけど、翻訳は単語単位でやらなければなりません……。
    for i in range(1, Y_MAXIMUM_POSITION):
        # 単語を翻訳します。
        ys[:, i] = np.argmax(model.predict((xs, ys[:, :i]), batch_size=256)[:, -1], axis=-1)  # 256並列で、次の単語を予測します。対象以外の単語も予測されますけど、無視します。

        # 文の終了記号が出力されたか確認します。
        is_ends |= ys[:, i] == ENCODES['$']

        # すべての文の翻訳が完了した場合はループをブレークします。
        if np.all(is_ends):
            break

    return ys
