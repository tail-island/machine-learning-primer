# TensorFlowで深層学習

LightGBMで勾配ブースティングが終わりましたので、お待ちかねの深層学習です。

でね、深層学習と言われて思い浮かべるだろう以下の図のモデルは全結合（dense）と呼ばれる層を重ねたもので、実は、そんなに精度が高くありません。

![Multilayer Perceptron](./image/multilayer-perceptron.png)

というわけで、まずは、このあまり精度が高くなかった全結合から始まった、深層学習の歴史を。

## 畳み込みから始まって

深層学習の精度が大きく向上したのは、畳み込み（convolution）という手法が発明されてからです。で、この畳み込みってのは、画像処理でいうところのフィルタリングそのものなんですよ。

画像においては、あるピクセルだけじゃなくて、そのピクセルの上下左右やそのさらに外側のピクセルとの関係が重要ですよね？　白い中に黒いピクセルが縦に並んでるから縦線と認識できるわけで。このような周りとの関係を汎用的な手法で抽出するのがフィルタリングです。

フィルタリング処理の具定例を示しましょう。こんな感じ。

~~~python
import cv2
import matplotlib.pyplot as plot
import numpy as np


def filter(image, kernel):
    result = np.zeros((np.shape(image)[0] - 2, np.shape(image)[1] - 2))

    for y in range(np.shape(result)[0]):
        for x in range(np.shape(result)[1]):
            # 画像の該当部分とカーネルのベクトルの内積を求めて新たな画像を作成します。
            result[y, x] = np.inner(image[y: y + 3, x: x + 3].flatten(), kernel.flatten())

    result[result <   0] =   0  # noqa: E222
    result[result > 255] = 255

    return result


image = cv2.imread('4.2.07.tiff')

plot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plot.show()

plot.imshow(filter(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), np.array([[0, 2, 0], [2, -8, 2], [0, 2, 0]])))
plot.show()
~~~

ある点の周囲の画像と行列（コード中の`kernel`）をベクトルに変換（`flatten()`）して内積（`np.inner()`）して、新たな画像を作るわけですね。たったこれだけで、画像から輪郭を抽出することができます。

![元画像](./image/4.2.07.png)

![フィルタリングの結果](./image/filtering.png)

で、本稿は文系のための文書ですから、ベクトルの内積について補足しましょう。数式は嫌なので、図でやります。題材は転職で。転職先を、給料と仕事の楽しさで表現します。

で、あれもこれも全部欲しいというのは贅沢すぎるので、転職先の評価軸は、距離が1のベクトルで表現するとしましょう。下の図の、左下から円周上のどこかに向かうベクトルが、転職先の評価軸となるわけですね。

![転職先を給料と仕事の楽しさでプロット](./image/offers.png)

で、「高い給料をもらえるなら非合法な仕事でもオッケー」とか「仕事が楽しければ霞を食べて生きていける」という特殊な場合はそれぞれX軸とY軸の値を見るだけでその軸と垂直なもう片方の軸の値を無視できるのですけど、普通は、給料2割で仕事の楽しさ8割とかで評価したいですよね？　そんな時は、ベクトルの内積が役に立ちます。

![転職先を評価ベクトルと内積で評価](./image/evaluate_offers_with_inner_product.png)

上の図のように、ベクトルの内積というのは、あるベクトルの視点で、他のベクトルがどの程度の量になるのかを表現しす。赤色で描かれた給料2割で仕事の楽しさ8割のベクトル上での、各転職先の大きさはどのくらいなのかがこれで一目瞭然で、転職がはかどっちゃうこと請け合い。しかも、ベクトルの内積ってのは2次元の画像だけじゃなくて、3次元でも4次元でも、1,024次元とかでも成り立つんですよ。だから、先ほどのコードでの`kernel`のような、9次元のベクトルでも使えるんです。

でね、先ほどのプログラムの`kernel`の値は、輪郭の場合に大きな値になるような行列になっていたんですよ。輪郭抽出なんて機械学習に関係なさそうと思ったかもしれませんけど、もし、こんな感じに丸くなっているとか、こんな感じに尖っているとか、こんな感じに交差しているとかを表現するベクトルを作るなら、しかもそれが大量にあるならば、たとえば文字認識とかを高い精度でできると思いませんか？

深層学習は機械学習なので、どのような`kernel`を使うと文字認識に有効なのかとか、犬と猫を区別するにはどのような`kernel`があればよいのかとかは、コンピューターが調整してくれます。とても楽ちんで、しかも精度が高い！

と、これが畳み込みで、この畳み込みのおかげで深層学習での画像認識の精度は大幅に向上したんです。

## 今は、アテンション

でもね、畳み込みって隣り合うものとの関係しか抽出できないので、自然言語のような遠くのものとの関係がある場合には使いづらかったんですよ。そんな場合にも対応できるように編み出されたのがアテンションです。

たとえば、前の文章の「そんな場合」は遠くに位置する「自然言語のような遠くのものとの関係がある場合」で、ほら、自然言語処理では遠くにあるものと関係があるでしょ？

自然言語処理では、単語をベクトルで表現します。「私」＝[0.1, 0.0, 0.4, 0.5, 0.0]みたいな感じ。で、単語のベクトルを要素に持つ行列と単語のベクトルの内積をとる（内積attentionの場合。他に加法attentionってのもありますけど私は使ったことない）と各単語の重みが出てきて、で、その重みに合わせて単語ベクトルと内積をとって、単語と単語の関係を考慮した行列を作成します。で、これだけだとただの行列計算なので、機械学習できるよう、重みパラメータを追加したのがアテンションです。

あとは、深層学習ってのは本来固定長の入力しか受け付けられないのだけど、それを数式上の工夫で可変長にして、翻訳で精度を大幅に向上させたのがTransformerという深層学習のモデルです。

## 使用するライブラリは、TensorFlow

大雑把な歴史の話が終わりましたので、使用するライブラリを選んでいきましょう。玄人はLuaでTorchかPythonでPyTorchを使うみたいで、深層学習の論文ではこれらの環境が良く使われているみたいです。が、素人の私は敢えてGoogle社のTensorFlowを推薦します。

論文の追試をするとか新しい論文を書くとかなら他の論文と同じ環境が良いのでしょうけど、実際の案件で使用する場合は、やっぱり寄らば大樹の影ですよ。TensorFlowは、様々なプログラミング言語から利用できたり、スマートフォンやIoTデバイスで実行できたりと、周辺機能の充実具合が段違いで優れているので将来のビジネス展開もバッチリですぜ。

## TensorFlowでTransformer

では、いきなりで申し訳ないのですけど、流行りのTransformerをTensorFlowで作ります。

いきなりそんな無茶なと感じた貴方は、[言語理解のためのTransformerモデル](https://www.tensorflow.org/tutorials/text/transformer?hl=ja)を開いてみてください。これ、TensorFlowのチュートリアルの一部なんですけど、ここに懇切丁寧にTransformerの作り方が書いてあります。

チュートリアルというのは、コンピューター業界では手を動かして実際にプログラムを作ることや、入門編という意味で使用されます。TensorFlowのチュートリアルはたぶん両方の意味で、入門レベルの人にも分かる記述で、実際に手を動かしてプログラムを作っている間に嫌でも理解できちゃうという内容になっています。いきなりTransformerは難しそうという場合は、[はじめてのニューラルネットワーク：分類問題の初歩](https://www.tensorflow.org/tutorials/keras/classification?hl=ja)から順を追ってやっていけばオッケーです。

ただね、TensorFlowを作っているような人ってのはとても頭が良い人で、で、頭が良い人ってのはたいていプログラミングが下手なんですよね……。彼らは頭が良いので、複雑なものを複雑なままで理解できます。で、理解した複雑な内容をそのまま複雑なプログラムとして実装しちゃう。ビジネス・アプリケーションのプログラマーが考えているようなリーダビリティへの考慮とかはゼロの、複雑怪奇な糞コードを書きやがるんですよ。

だから、TensorFlowのチュートリアルをやったあとは、できあがったコードをリファクタリングしましょう。私がリファクタリングした結果は、こんな感じなりました。

~~~python
import numpy as np
import tensorflow as tf

from funcy import func_partial, rcompose


def transformer(num_blocks, d_model, num_heads, d_ff, x_vocab_size, y_vocab_size, x_maximum_position, y_maximum_position, dropout_rate):
    # KerasやTensorflowのレイヤーや関数をラップします。

    def dense(units):
        return tf.keras.layers.Dense(units)

    def dropout(rate):
        return tf.keras.layers.Dropout(rate)

    def embedding(input_dim, output_dim):
        return tf.keras.layers.Embedding(input_dim, output_dim)

    def layer_normalization():
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def relu():
        return tf.keras.layers.ReLU()

    def reshape(target_shape):
        return tf.keras.layers.Reshape(target_shape)

    def transpose(perm):
        return func_partial(tf.transpose, perm=perm)

    # Transformerに必要な演算を定義します。

    def scaled_dot_product_attention(x):
        query, key, value, mask = x

        return tf.matmul(tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32)) + mask * -1e9, axis=-1), value)

    def multi_head_attention(d_model, num_heads):
        split  = rcompose(reshape((-1, num_heads, d_model // num_heads)),  # noqa: E221
                          transpose((0, 2, 1, 3)))
        concat = rcompose(transpose((0, 2, 1, 3)),
                          reshape((-1, d_model)))

        def op(inputs):
            q, k, v, mask = inputs

            o = scaled_dot_product_attention((split(dense(d_model)(q)),
                                              split(dense(d_model)(k)),
                                              split(dense(d_model)(v)),
                                              mask))
            o = concat(o)
            o = dense(d_model)(o)

            return o

        return op

    def point_wise_feed_forward(d_model, d_ff):
        return rcompose(dense(d_ff),
                        relu(),
                        dense(d_model))

    def encoder_block(d_model, num_heads, d_ff, dropout_rate):
        def op(inputs):
            x, mask = inputs

            o = layer_normalization()(dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((x, x, x, mask))) + x)
            o = layer_normalization()(dropout(dropout_rate)(point_wise_feed_forward(d_model, d_ff)(o)) + o)

            return o

        return op

    def decoder_block(d_model, num_heads, d_ff, dropout_rate):
        def op(inputs):
            y, y_mask, z, z_mask = inputs

            o = layer_normalization()(dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((y, y, y, y_mask))) + y)
            o = layer_normalization()(dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((o, z, z, z_mask))) + o)
            o = layer_normalization()(dropout(dropout_rate)(point_wise_feed_forward(d_model, d_ff)(o)) + o)

            return o

        return op

    def get_positional_encoding(maximum_position, d_model):
        result = np.empty((maximum_position, d_model), dtype=np.float32)

        angles = np.arange(maximum_position)[:, np.newaxis] / np.power(10000, 2 * np.arange(d_model // 2) / d_model)

        result[:, 0::2] = np.sin(angles)  # 偶数はsin
        result[:, 1::2] = np.cos(angles)  # 奇数はcos
        result = tf.cast(result[np.newaxis, ...], dtype=tf.float32)

        return result

    def encoder(num_blocks, d_model, num_heads, d_ff, vocab_size, maximum_position, dropout_rate):
        normalize_factor = tf.math.sqrt(tf.cast(d_model, tf.float32))
        positional_encoding = get_positional_encoding(maximum_position, d_model)

        def op(inputs):
            x, mask = inputs

            o = dropout(dropout_rate)(embedding(vocab_size, d_model)(x) * normalize_factor + positional_encoding[:, :tf.shape(x)[1], :])

            for _ in range(num_blocks):
                o = encoder_block(d_model, num_heads, d_ff, dropout_rate)((o, mask))

            return o

        return op

    def decoder(num_blocks, d_model, num_heads, d_ff, vocab_size, maximum_position, dropout_rate):
        normalize_factor = tf.math.sqrt(tf.cast(d_model, tf.float32))
        positional_encoding = get_positional_encoding(maximum_position, d_model)

        def op(inputs):
            y, y_mask, z, z_mask = inputs

            o = dropout(dropout_rate)(embedding(vocab_size, d_model)(y) * normalize_factor + positional_encoding[:, :tf.shape(y)[1], :])

            for _ in range(num_blocks):
                o = decoder_block(d_model, num_heads, d_ff, dropout_rate)((o, y_mask, z, z_mask))

            return o

        return op

    def get_padding_mask(x):
        return tf.cast(tf.math.equal(x, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def get_look_ahead_mask(size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    def op(inputs):
        x, y = inputs

        o = encoder(num_blocks, d_model, num_heads, d_ff, x_vocab_size, x_maximum_position, dropout_rate)((x, get_padding_mask(x)))
        o = decoder(num_blocks, d_model, num_heads, d_ff, y_vocab_size, y_maximum_position, dropout_rate)((y, tf.maximum(get_look_ahead_mask(tf.shape(y)[1]), get_padding_mask(y)), o, get_padding_mask(x)))
        o = dense(y_vocab_size)(o)

        return o

    return op


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.d_model ** -0.5 * tf.math.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)


class Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Loss, self).__init__()

        self.sparse_categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, y_true, y_pred):
        return tf.reduce_mean(self.sparse_categorical_crossentropy(y_true, y_pred) * tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32))
~~~

上のコードの解説はTensorFlowのチュートリアルに任せることにして、なぜ誰かのTransformer実装を使わずに自前でTransformerを実装したのかを述べさせてください。

既存のTransformerの実装が存在しないわけじゃありません。GitHubを検索すれば大量に見つかるでしょう。それらを利用すれば、わざわざ実装せずともTransformerできます。でも、できればTensorFlowのチュートリアルを読んで、自前で実装して動かしてみていただきたいです。

その理由は、深層学習がいまだ未成熟な技術だから。新しいモデルが次々に出てきていて、そのモデルを使えば今までできなかったことができるようになって、ビジネスに大いに役立つかもしれません。その新しいモデルの実装があるとは限らないわけで、自前で実装しなければならないかもしれません。その時は、そのひとつ前のモデルの実装経験が役立つでしょう。

あと、深層学習は入力や出力の型が決まっているという問題もあります。画像認識のような場合はこれが結構痛くて、画像の大きさをモデルが要求する大きさに合わせたりしなければなりません。自前で作成するなら、モデルの中身を調整できるのでこんな問題はありません。

ただし、事前学習が必要なくらいに大きなニューラル・ネットワークの場合は、既存の実装を再利用するしかないですけどね……。まぁ、その場合でも自前で深層学習を実装したことによる深い理解は役に立つんじゃないかと。

というわけで実装したTransformerを使用して機械学習してみましょう。TensorFlowのチュートリアルではポルトガル語から英語への翻訳をやっていますけど、ポルトガル語も英語も分からないので、今回は足し算と引き算でやります。`1 + 1`をTransformerにかけたら`2`が出力されるわけですね。

データセットは、以下のようになっています。

~~~csv
Id,Expression,Answer
0,934+952,1886
1,68+487,555
2,550+690,1240
3,360+421,781
4,214+844,1058
5,453+728,1181
6,798+178,976
7,199+809,1008
8,182+317,499
9,818+788,1606
10,966+380,1346
~~~

このデータセットを読み込むモジュール（dataset.py）はこんな感じ。

~~~python
import numpy as np
import os.path as path
import pandas as pd

from funcy import concat, count, dropwhile, map, take, takewhile


# 使用される単語。自然言語処理の処理単位は単語です。今回は面倒なので、文字単位にしました。
WORDS = tuple(concat((' ',), ('+', '-'), map(str, range(10)), ('^', '$')))

# 単語を整数にエンコード/デコードするためのdictです。
ENCODES = dict(zip(WORDS, count()))
DECODES = dict(zip(count(), WORDS))


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', filename), dtype={'Expression': 'string', 'Answer': 'string'})


# 訓練用のDataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用のDataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 深層学習するために、文をエンコードして数値の集合に変換します。
def encode(sentence, max_sentence_length):
    return take(max_sentence_length + 2, concat((ENCODES['^'],),  # 文の開始
                                                map(ENCODES, sentence),
                                                (ENCODES['$'],),  # 文の終了
                                                (ENCODES[' '],) * max_sentence_length))  # 残りは空白で埋めます。長さを揃えないと、深層学習できないためです。


# 深層学習が出力した数値の集合を、デコードして文字列に変換します。
def decode(encoded):
    return ''.join(takewhile(lambda c: c != '$', dropwhile(lambda c: c == '^', map(DECODES, encoded))))


# 入力データを取得します。
def get_xs(data_frame):
    strings = data_frame['Expression']
    max_length = max(map(len, strings))

    return np.array(tuple(map(lambda string: tuple(encode(string, max_length)), strings)), dtype=np.int64)


# 正解データを取得します。
def get_ys(data_frame):
    strings = data_frame['Answer']
    max_length = max(map(len, strings))

    return np.array(tuple(map(lambda string: tuple(encode(string, max_length)), strings)), dtype=np.int64)
~~~

自然言語処理では単語単位に一意のIDを割り振る必要があるので、そのためのdictを作成したり文字列をID列にエンコードしたりID列を文字列にデコードしたりする処理を作成しています。

Transformerのハイパー・パラメーターはこんな感じ。

~~~python
from dataset import WORDS


NUM_BLOCKS = 3             # 簡単なタスクなので、Attention is all you needの半分
D_MODEL = 256              # 簡単なタスクなので、Attention is all you needの半分
D_FF = 1024                # 簡単なタスクなので、Attention is all you needの半分
NUM_HEADS = 4              # 簡単なタスクなので、Attention is all you needの半分
DROPOUT_RATE = 0.1         # ここはAttention is all you needのまま
X_VOCAB_SIZE = len(WORDS)
Y_VOCAB_SIZE = len(WORDS)
X_MAXIMUM_POSITION = 20    # 余裕を持って多めに
Y_MAXIMUM_POSITION = 20    # 余裕を持って多めに
~~~

Transformerは[Attention is all you need](https://arxiv.org/abs/1706.03762)という論文のモデルで、この論文が使用しているハイパー・パラメーターをフィーリングで半分に減らしました。

機械学習する部分のコードは、こんな感じ。

~~~python
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

# データを読み込みます。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# 訓練データセットと検証データセットに分割するためのインデックスを作成します。
indices = rng.permutation(np.arange(len(xs)))

# 訓練データセットを取得します。
train_xs = xs[indices[2000:]]
train_ys = ys[indices[2000:]]

# 検証データセットを取得します。
valid_xs = xs[indices[:2000]]
valid_ys = ys[indices[:2000]]

# Transformerを作成します。
op = transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, X_VOCAB_SIZE, Y_VOCAB_SIZE, X_MAXIMUM_POSITION, Y_MAXIMUM_POSITION, DROPOUT_RATE)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)([tf.keras.Input(shape=(None,)), tf.keras.Input(shape=(None,))]))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=Loss(), metrics=())
# model.summary()

# 機械学習して、モデルを保存します。
model.fit((train_xs, train_ys[:, :-1]), train_ys[:, 1:], batch_size=256, epochs=100, validation_data=((valid_xs, valid_ys[:, :-1]), valid_ys[:, 1:]))
model.save('model', include_optimizer=False)

# 実際に予測させて、精度を確認します。
print(f'Accuracy = {sum(starmap(eq, zip(map(decode, valid_ys), map(decode, translate(model, valid_xs))))) / len(valid_xs)}')
~~~

TensorFlowのTransformerのチュートリアルでは、TensorFlowの生APIを使って細かく学習を制御しているのですけど、生APIを使うのは面倒なので簡単便利ライブラリであるKeras（TensorFlowに付属しています）を使用しました。`fit()`の時にいろいろ情報を出力してくれて面白いですよ。

で、上のコード中で使用している`translate()`は、後述するテストのモジュールでも使用するので、共用できるようにモジュール化しました。

~~~python
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
~~~

Transformerは、学習は文の単位でやる（そのために、文の先を参照しないようにマスクを設定している）のですけど、翻訳するときの単位は単語です。元の文＋1単語目までを入力に2単語目を出力するわけですね。実は3つ目の単語を予測する時にも2つめの単語を予測して出力しているのですけど、`[:, -1]`で捨てています。計算の無駄な気がしますけど、効率よく学習するためなので我慢してください。

最後。テスト・データを翻訳させて、精度を確認します。コードはこんな感じ。

~~~python
import tensorflow as tf

from dataset import decode, get_test_data_frame, get_xs, get_ys
from translation import translate


# モデルを取得します。
model = tf.keras.models.load_model('model')

# データを取得します。
data_frame = get_test_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# 正解した数。
equal_count = 0

# 実際に予測させて、正解した数を取得します。
for x, y_true, y_pred in zip(xs, ys, translate(model, xs)):
    y_true_string = decode(y_true)
    y_pred_string = decode(y_pred)

    equal = y_true_string == y_pred_string
    equal_count += equal

    print(f'{decode(x)} {"==" if equal else "!="} {y_pred_string}')  # ついでなので、予測結果を出力させます。

# 精度を出力します。
print(f'Accuracy: {equal_count / len(xs)}')
~~~

さて、その精度はどうなったかというと、0.947という高い精度になりました！　大量データを用意すれば、Transformerで翻訳できそうですね。チュートリアルがあるので、作るのはそんなに大変じゃないですし。

## 深層学習の使いどころ

……でも、ちょっと待って。足し算とか引き算なら、深層学習を使わないで実際に足し算とか引き算するコードを書けば精度が100%になるのでは？

はい。おっしゃる通り。もし入力と出力の関係を定義できて、それをシミュレーションできるなら、深層学習を使わないでシミュレーターを実装すれば良いでしょう。シミュレーターを作れるなら、シミュレーターの方が精度が高そうですもんね。

ただ、シミュレーターを作成可能な場合であっても深層学習が役に立つ場合もあるんですよ。それがどんな場合かというと、シミュレーションにやたらと時間がかかる場合です。実は深層学習はすべての関数を近似できるらしくて（証明は難しくて理解できなかったけど）、ということは、シミュレーションを近似できちゃうというわけ。

たとえば原子の動きのシミュレーションは、量子力学の基本法則に立脚した第一原理計算というのがあって、それを活用する密度汎関数法（density functional theory）という計算でできる（らしい）んです。ただ、これってとにかく計算に時間がかかる（っぽい）んですよ。複雑な原子のシミュレーションはとにかく時間がかかってやってられないので、だからこれを深層学習で近似しちゃおうというのが、Preferred Networks社とENEOS社の[Matlantis](https://matlantis.com/ja/)で、マテリアル・インフォマティクス分野ではとても役に立つ（みたい）です。

こんな感じで、もし遅くてやってられないシミュレーターがあるなら、深層学習で代替できないかを検討してみるのも良いと思います。

あ、シミュレーターを作れるかもしれないけど作るのが面倒なのでとりあえず深層学習ってのもアリで、私はこっち派だったりします。

# Vision Transformerで画像のクラス分類

話を元に戻して、KaggleのGetting Startedの[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)で画像のクラス分類をやりましょう。手書き数字の画像認識問題ですね。

## Vision Transformer？

Vision Transformerは、翻訳で大成功したTransformerを元にして作成された、様々な自然言語処理タスクで人間を超えたと話題になったBERTの、画像認識版です。BERTはTransformerを少し修正するだけで作成できるのですけど、Vision Transformerも同様にTransformerの修正で作成可能です。

TensorFlowのTransformerのチュートリアルをやった皆様はすでにご存じだと思いますけど、Transformerはエンコーダーとデコーダーで構成されています。エンコーダーで文章全体を表現する行列を作成して、その行列をデコーダーで変換して翻訳先の文章を作るわけ。BERTやVision Transformerはこのエンコーダーだけを使用します。

BERTやVision Trasformerでは、文章の頭にクラスを表現するベクトルを追加して、で、そのベクトルに相当する部分の出力を使用してクラス分類する（BERTの場合は、ネガ/ポジの感情判定したりする）わけですね。BERTの場合は、入力となった単語のベクトルに相当する部分を使って、文と質問のペアの、質問に相当する文の箇所を推論したりもします。

で、BERTはその学習方法が面白かったりする（簡単にいくらでも作れる問題で事前学習（pre training）している）のですけど、詳細はBERTと事前学習で検索してみてください。これは説明責任を放棄しているわけではなくて、流行りの技術である深層学習はいろいろな人が解説を書いてくれていて、検索するだけで大量の解説が手に入ることを知って欲しいから。

これはVision Transformerについても同様で、検索すれば大量の分かりやすい解説が見つかります。解説が間違えている可能性はあるので[論文](https://arxiv.org/abs/2010.11929)の参照は必要ですけど、論文だけで理解するより遥かに楽ちんです。

最新の深層学習のモデルだと論文を読むしかないので深層学習は難しいのですけど、半年から一年遅れくらいでよいなら、大量の解説を利用できるので深層学習はそんなに難しくないですよ。

## Vision Transformerを実装する

ありがたいことにVision Transformerの[Keras公式実装](https://keras.io/examples/vision/image_classification_with_vision_transformer/)もあるので、これを参考にすれば簡単に実装できます。今回は、論文から少し離れて、Kerasの実装に合う形（クラスを先頭に追加するのではなく、全結合層でクラス分類する等）で実装しました。

というわけで、コードはこんな感じになりました。

~~~python
import numpy as np
import tensorflow as tf

from funcy import func_partial, rcompose


def vision_transformer(num_blocks, d_model, num_heads, d_ff, y_vocab_size, x_maximum_position, dropout_rate):
    # KerasやTensorflowのレイヤーや関数をラップします。

    def dense(units):
        return tf.keras.layers.Dense(units)

    def dropout(rate):
        return tf.keras.layers.Dropout(rate)

    def embedding(input_dim, output_dim):
        return tf.keras.layers.Embedding(input_dim, output_dim)

    def flatten():
        return tf.keras.layers.Flatten()

    def gelu():
        return tf.keras.activations.gelu

    def layer_normalization():
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def reshape(target_shape):
        return tf.keras.layers.Reshape(target_shape)

    def softmax():
        return tf.keras.layers.Softmax()

    def transpose(perm):
        return func_partial(tf.transpose, perm=perm)

    # Transformerに必要な演算を定義します。

    def scaled_dot_product_attention(x):
        query, key, value = x

        return tf.matmul(tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32)), axis=-1), value)

    def multi_head_attention(d_model, num_heads):
        split  = rcompose(reshape((-1, num_heads, d_model // num_heads)),  # noqa: E221
                          transpose((0, 2, 1, 3)))
        concat = rcompose(transpose((0, 2, 1, 3)),
                          reshape((-1, d_model)))

        def op(inputs):
            q, k, v = inputs

            o = scaled_dot_product_attention((split(dense(d_model)(q)),
                                              split(dense(d_model)(k)),
                                              split(dense(d_model)(v))))
            o = concat(o)
            o = dense(d_model)(o)

            return o

        return op

    def point_wise_feed_forward(d_model, d_ff):
        return rcompose(dense(d_ff),
                        gelu(),
                        dense(d_model))

    def encoder_block(d_model, num_heads, d_ff, dropout_rate):
        def op(inputs):
            x = inputs

            o = layer_normalization()(x)
            o = dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((o, o, o))) + o
            o = layer_normalization()(o)
            o = dropout(dropout_rate)(point_wise_feed_forward(d_model, d_ff)(o)) + o

            return o

        return op

    def get_positional_encoding(maximum_position, d_model):
        result = embedding(maximum_position, d_model)(tf.range(0, maximum_position))

        return result[np.newaxis, ...]

    def encoder(num_blocks, d_model, num_heads, d_ff, maximum_position, dropout_rate):
        normalize_factor = tf.math.sqrt(tf.cast(d_model, tf.float32))
        positional_encoding = get_positional_encoding(maximum_position, d_model)

        def op(inputs):
            x = inputs

            o = dropout(dropout_rate)(x * normalize_factor + positional_encoding)

            for _ in range(num_blocks):
                o = encoder_block(d_model, num_heads, d_ff, dropout_rate)((o))

            return o

        return op

    def op(inputs):
        x = inputs

        return softmax()(dense(y_vocab_size)(flatten()(encoder(num_blocks, d_model, num_heads, d_ff, x_maximum_position, dropout_rate)(x))))

    return op


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.d_model ** -0.5 * tf.math.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)
~~~

自然言語処理では単語を埋め込み（embedding）してベクトル化して入力にするのですけど、Vision Transformerでは画像を例えば16×16のパッチに分けて、それぞれのパッチをベクトルとして扱って入力としているのがキモです（だから、`encoder()`の最初の`embedding`がなくなっています）。

あとは、デコーダーが不要だったり、先読みが無関係なのでマスクがいらなかったり、Layer Normalizationが先だったりReLUの代わりにGERUを使ったりと、少しだけニューラル・ネットワークが違うくらい。Transformerのコードがあるなら、その修正で簡単に作成できます。

……とはいっても、いざ修正してみると大量のエラーが出たりするんですけどね。でも、安心してください。エラーが出るのは関数の引数の型が合わないからで、しかもその型ってのは行列の形だったりします。普段のプログラミングで様々な型を考慮しなければならないのに比べれば、考えなければならないことが少ない分だけ簡単です。`print(tf.shape(x))`とかすれば行列の形が表示されますしね。

## データセットを取得する

Vision Transformerが出来たので、次は、データセットを取得するモジュールを作成します。

~~~python
import numpy as np
import pandas as pd
import os.path as path

from params import D_MODEL_HEIGHT, D_MODEL_WIDTH


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', 'digit-recognizer', filename))


# 訓練用DataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用DataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 画像をパッチに分割します。
def encode(image):
    def impl():
        for i in range(28 // D_MODEL_HEIGHT):
            for j in range(28 // D_MODEL_WIDTH):
                yield image[i * D_MODEL_HEIGHT: i * D_MODEL_HEIGHT + D_MODEL_HEIGHT, j * D_MODEL_WIDTH: j * D_MODEL_WIDTH + D_MODEL_WIDTH].flatten()

    return np.array(tuple(impl()))


# パッチを画像に戻します。
def decode(encoded):
    result = np.zeros((28, 28), dtype=np.float32)

    for i in range(28 // D_MODEL_HEIGHT):
        for j in range(28 // D_MODEL_WIDTH):
            result[i * D_MODEL_HEIGHT: i * D_MODEL_HEIGHT + D_MODEL_HEIGHT, j * D_MODEL_WIDTH: j * D_MODEL_WIDTH + D_MODEL_WIDTH] = np.reshape(encoded[i * 28 // D_MODEL_WIDTH + j], (D_MODEL_WIDTH, D_MODEL_HEIGHT))

    return result


# 入力データを取得します。
def get_xs(data_frame):
    return np.array(tuple(map(encode, np.reshape(data_frame[list(map(lambda i: f'pixel{i}', range(784)))].values / 255, (-1, 28, 28)))))


# 出力データを取得します。
def get_ys(data_frame):
    return data_frame['label'].values
~~~

KerasのVision Transformer実装では画像をパッチに分割する処理がニューラル・ネットワークに含まれていたのですけど、今回は、今後の解説の都合によりデータセット読み込み時にパッチに分割しました（ごめんなさい。後で述べるデータの水増しができないので少々不利なやり方です……）。

あと、`get_xs()`の中で255で割っているのは、データ中は0～255になっているのを0～1に正規化するためです。深層学習のハイパー・パラメーターはデータが正規化されていることを前提にしているので、0～255のような大きな値だと学習が正常に進まないんですよ。

## ハイパー・パラメーターを設定する

上のデータセットのモジュールでも使用していたパッチの大きさ（`D_MODEL_WIDTH`と`D_MODEL_HEIGHT`）を含むハイパー・パラメーターを設定するモジュールを作成します。

~~~python
NUM_BLOCKS = 3
D_MODEL_WIDTH = 4
D_MODEL_HEIGHT = 4
D_MODEL = D_MODEL_WIDTH * D_MODEL_HEIGHT
D_FF = 1024
NUM_HEADS = 4
DROPOUT_RATE = 0.1
Y_VOCAB_SIZE = 10
~~~

パッチの大きさは4×4ドットにしました。今回のデータは28×28ドットなので、これなら割り切れて、あと、そこそこパッチ数も大きくなるんじゃないかと。他は調整が面倒だったのでTransformerを作ったときの値から変えていません。

## 機械学習する

準備が整ったので、機械学習しましょう。コードはこんな感じ。

~~~python
import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from itertools import starmap
from operator import eq
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, DROPOUT_RATE
from vision_transformer import LearningRateSchedule, vision_transformer


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# 訓練データセットと検証データセットに分割するためのインデックスを作成します。
indices = rng.permutation(np.arange(len(xs)))

# 訓練データセットを取得します。
train_xs = xs[indices[2000:]]
train_ys = ys[indices[2000:]]

# 検証データセットを取得します。
valid_xs = xs[indices[:2000]]
valid_ys = ys[indices[:2000]]

# Vision Transformerを作成します。
op = vision_transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, 28 * 28 // D_MODEL, DROPOUT_RATE)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

# 機械学習します。
model.fit(train_xs, train_ys, batch_size=256, epochs=10, validation_data=(valid_xs, valid_ys))

# 精度を表示します。
print(f'Accuracy = {sum(starmap(eq, zip(valid_ys, np.argmax(model.predict(valid_xs, batch_size=256), axis=-1)))) / len(valid_xs)}')
~~~

うん、Transformerの時とほぼ同じですね。Transformerの時は入力の行列の形を事前に決められなかった（文の長さは事前には分からない）ので変な形になっていました（それでも問題ないようにニューラル・ネットワークが工夫されていた）けど、今回は画像の大きさが決まっているので`tf.keras.Input(shape=np.shape(xs)[1:])`として行列の形を指定しました。

あとは、`compile()`の時の引数が`loss='sparse_categorical_crossentropy'`となっています。これは、クラス分類をする時用の損失関数（誤差を計算するための関数。LightGBMの`metric`）です。あと、Kerasでは、損失関数とは別に、人間に分かりやすい測定結果も出力できるようになっていて、それが`metrics`です。今回は、ここに精度である`accuracy`を指定しました。

残りはTransformerの時と同じです。実行してみると、精度は0.963でした。あれ、なんか精度低い？　2,000人くらい中の1,600番目くらい？

## 予測モデルを作成する

精度を確認するために、予測モデルを作成してテスト・データで予測して、Kaggleに提出してみましょう。まずは、予測モデルの作成から。

~~~python
import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from params import NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, DROPOUT_RATE
from vision_transformer import LearningRateSchedule, vision_transformer


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# Vision Transformerを作成します。
op = vision_transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, Y_VOCAB_SIZE, 28 * 28 // D_MODEL, DROPOUT_RATE)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

# 機械学習して、モデルを保存します。
model.fit(xs, ys, batch_size=256, epochs=100)
model.save('digit-recognizer-model', include_optimizer=False)
~~~

機械学習するモジュールからコードを切り貼りして、最後にモデルを保存する処理を追加しただけです。あ、精度が上がるかもと思って、エポック数は10から100に増やしました。

## 解答を作成する

テスト・データで予測して解答を作成するモジュールを作成します。コードはこんな感じ。

~~~python
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import get_test_data_frame, get_xs


# モデルを取得します。
model = tf.keras.models.load_model('digit-recognizer-model')

# データを取得します。
data_frame = get_test_data_frame()

# 入力データを取得します。
xs = get_xs(data_frame)

# 解答を作成して、保存します。
submission = pd.DataFrame({'ImageId': data_frame.index + 1, 'Label': np.argmax(model.predict(xs, batch_size=256), axis=-1)})
submission.to_csv('submission.csv', index=False)
~~~

さて、テスト・データでの精度は、0.98232。エポック数を増やした分だけ精度が上がったような気がしますが、それでもやっぱり2,000人くらい中の900番目くらい。Digit Recognizerでもやっぱりカンニングしている連中はいるけど、それにしたって低すぎます。2020年に画像認識の革命と大騒ぎになったVision Transformerを使っているのにどういうこと？

……ここまで真面目に読んでくださった皆様ごめんなさい。たぶんこんな結果になること、実は知ってた。

というのも、Transformer系って、大量のデータがないと精度が高くならないんですよ。Transformerの後に自然言語処理で人間を超えたと話題になったBERTはBooksCorpusという8億ワードのデータとWikipediaの25億ワードのデータで事前学習しましたし、Vision Transformerで最高の精度を出したときはGoogle社のプライベートなデータセットであるJFT-300Mという3億枚の画像を使用して事前学習しています。データの単位は億なんですな。

それと比較して今回は、事前学習がゼロで、学習に使用したデータは4万2千……。そりゃ、精度なんか出ませんよね。

こんな感じで、実用という意味では、深層学習の最先端は我々一般人の手が届かない遥か彼方に行ってしまいました。そもそもデータが集まらないし、仮にデータが集まっても一般人の手に入るレベルの計算リソースだととても学習させられないし。

というわけで、事前学習済みのモデルに転移学習（fine tuning）するような場合にしか使えないんじゃないかなぁと。貴方の目的に合う事前学習と入力のものが運良く見つかったならば、ですけど。

# DenseNetで画像のクラス分類

だから今回は、アテンションではなく、畳み込みを使うモデルで勝負しました。DenseNetです。

## DenseNetを実装する

DenseNetの実装はこんな感じ。

~~~python
import tensorflow as tf

from funcy import rcompose


def dense_net(growth_rate, classes):
    # KerasやTensorflowのレイヤーや関数をラップします。

    def average_pooling_2d(pool_size, strides=None):
        return tf.keras.layers.AveragePooling2D(pool_size, strides=strides)

    def concatenate():
        return tf.keras.layers.Concatenate()

    def batch_normalization():
        return tf.keras.layers.BatchNormalization(epsilon=1.001e-5)

    def conv_2d(filters, kernel_size):
        return tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)

    def dense(units):
        return tf.keras.layers.Dense(units)

    def global_average_pooling_2d():
        return tf.keras.layers.GlobalAveragePooling2D()

    def relu():
        return tf.keras.layers.ReLU()

    def softmax():
        return tf.keras.layers.Softmax()

    def zero_padding2d(padding):
        return tf.keras.layers.ZeroPadding2D(padding=padding)

    # DenseNetに必要な演算を定義します。

    def dense_block(blocks):
        def op(inputs):
            result = inputs

            for _ in range(blocks):
                result_ = batch_normalization()(result)
                result_ = relu()(result_)
                result_ = conv_2d(4 * growth_rate, 1)(result_)
                result_ = batch_normalization()(result_)
                result_ = relu()(result_)
                result_ = conv_2d(growth_rate, 3)(result_)

                result = concatenate()((result, result_))

            return result

        return op

    def transition_block():
        def op(inputs):
            result = batch_normalization()(inputs)
            result = relu()(result)
            result = conv_2d(int(tf.keras.backend.int_shape(inputs)[3] * 0.5), 1)(result)
            result = average_pooling_2d(2, strides=2)(result)

            return result

        return op

    def dense_net_121():
        return rcompose(dense_block(6),
                        transition_block(),
                        dense_block(12),
                        transition_block(),
                        dense_block(24),
                        transition_block(),
                        dense_block(16))

    return rcompose(conv_2d(64, 1),
                    batch_normalization(),
                    relu(),  # ImageNetの場合はここでConv2DやMaxPooling2Dで画素数を減らすのですけど、今回はもともとの画素数が少ないのでやりません。
                    dense_net_121(),
                    batch_normalization(),
                    relu(),
                    global_average_pooling_2d(),
                    dense(classes),
                    softmax())
~~~

Transformerに比べるとだいぶん簡単です。例によって、検索で見つけた様々な解説と、[KerasのDenseNet実装](https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py)を参考にしました。

DenseNetはImageNetというデータセットを前提にしていて、224×224ドットの画像が入力されることを前提にしています。Digit Recognizerの入力は28×28ドットとだいぶん小さいので、コメントに書いたように前処理を省略しました。モデルを自作すると、このような調整が出来て便利です。

## データセットを取得する

データセットを取得するモジュールを作成します。

~~~python
import numpy as np
import pandas as pd
import os.path as path


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', 'digit-recognizer', filename))


# 訓練用DataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用DataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 入力データを取得します。
def get_xs(data_frame):
    return np.array(np.reshape(data_frame[list(map(lambda i: f'pixel{i}', range(784)))].values / 255, (-1, 28, 28, 1)))


# 出力データを取得します。
def get_ys(data_frame):
    return data_frame['label'].values
~~~

画像をパッチに分ける必要がなくなったので、Vision Transformerより簡単です。あ、255で割っているのは0～1に正規化するためで、正規化しないと学習が正常に進まないのはVision Transformerのところで述べた通りです。

## 機械学習する

準備ができたので、機械学習してみましょう。

~~~python
import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from dense_net import dense_net
from funcy import concat, identity, juxt, partial, repeat, take
from itertools import starmap
from operator import eq, getitem


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# 訓練データセットと検証データセットに分割するためのインデックスを作成します。
indices = rng.permutation(np.arange(len(xs)))

# 訓練データセットを取得します。
train_xs = xs[indices[2000:]]
train_ys = ys[indices[2000:]]

# 検証データセットを取得します。
valid_xs = xs[indices[:2000]]
valid_ys = ys[indices[:2000]]

# DenseNetを作成します。
op = dense_net(16, 10)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

batch_size = 128
epoch_size = 40

# データの水増しをします。
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=22.5,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2)

# 機械学習して、モデルを保存します。
model.fit_generator(image_data_generator.flow(train_xs, train_ys, batch_size=batch_size),
                    steps_per_epoch=len(train_xs) // batch_size,
                    epochs=epoch_size,
                    callbacks=(tf.keras.callbacks.LearningRateScheduler(partial(getitem, tuple(take(epoch_size, concat(repeat(0.01, epoch_size // 2), repeat(0.01 / 10, epoch_size // 4), repeat(0.01 / 100))))))),
                    validation_data=(valid_xs, valid_ys))

# 精度を出力します。
print(f'Accuracy = {sum(starmap(eq, zip(valid_ys, np.argmax(model.predict(valid_xs, batch_size=256), axis=-1)))) / len(valid_xs)}')
~~~

前半はVision Transformerの時と同じですが、後半はちょっと違っています。

まずは、Kerasのモデルを`compile()`するところ。Vision Transformerでは学習率を専用のクラスを作成して調整しましたが、今回は、`compile()`の時は`adam`を使うようにだけ指定して、後でコールバックを使用して学習率を調整しています。`fit()`や今回使用した`fit_generator()`の引数のcallbacksに`LearningRateScheduler`を指定して、そこでエポック数の半分はデフォルトの学習率、1/4はその1/10、残りは1/100にしています。

で、TransformerやVision Transformerでは曖昧にごまかした学習率の調整なのですけど、これ、遠くにあるゴール目指して歩く時の歩幅だと考えてみてください。まずは、ゴールが遠いので大股で歩かないとゴールにたどりつけませんから、歩幅を大きくします。でも、その歩幅のままだと、いつかはゴールを通り過ぎてしまいます。で、Uターンしたとしても、歩幅が大きいので、またゴールを通り過ぎてしまいます。それでは困るので、学習率を下げて歩幅を小さくします。これで、ゴールまでより近づくことができるようになりますが、その場合であっても、ゴールまでの距離より歩幅の方が大きければまた同じ問題が発生します。だから、念のためもう一度歩幅を狭めたというわけ。

次は、データの水増し（data augmentation）です。機械学習はデータに見つかるパターンしか学習しませんから、データ中のバリエーションを増やしたい。たとえば、少し右にずれている画像とか、少し回転している画像とか、あとは、今回は使えないですけど左右が反転した画像とか。犬猫認識で、もし左を向いたデータしかなければ、右を向いた犬は犬と識別できないことになっちゃいますもんね。というわけで、今回は`ImageDataGenerator`でデータを加工して水増ししました。上のコードでは、±22.5°まで傾けたり、2/10まで画像を左右や上下に動かしたりしています。

で、`ImageDataGenerator`のような、データがどんどん生成されるような場合（他に、データが大きすぎてメモリに乗らない場合にストレージから逐次的に読み込む場合なんてのもあります）は、これまでのように`fit()`ではなく`fit_generator()`を使用します。`fit()`とはちょっと引数が違うので気を付けてください。

以上でコードの解説は完了しましたので、実行して精度を測定します。で、精度は、0.9955でした。2,000人中の130位くらい。うん、やっぱり深層学習はこうでなきゃ。

## 予測モデルを作成する

テスト・データでもこのような高い精度がでるのか、確認しましょう。まずは、予測モデルの作成です。

~~~python
import numpy as np
import tensorflow as tf

from dataset import get_train_data_frame, get_xs, get_ys
from dense_net import dense_net
from funcy import concat, identity, juxt, partial, repeat, take
from operator import getitem


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# DenseNetを作成します。
op = dense_net(16, 10)

# Kerasのモデルを作成します。
model = tf.keras.Model(*juxt(identity, op)(tf.keras.Input(shape=np.shape(xs)[1:])))
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=('accuracy',))
# model.summary()

batch_size = 128
epoch_size = 40

# データの水増しをします。
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=22.5,
                                                                       width_shift_range=0.2,
                                                                       height_shift_range=0.2)

# 機械学習して、モデルを保存します。
model.fit_generator(image_data_generator.flow(xs, ys, batch_size=batch_size),
                    steps_per_epoch=len(xs) // batch_size,
                    epochs=epoch_size,
                    callbacks=(tf.keras.callbacks.LearningRateScheduler(partial(getitem, tuple(take(epoch_size, concat(repeat(0.01, epoch_size // 2), repeat(0.01 / 10, epoch_size // 4), repeat(0.01 / 100))))))))
model.save('digit-recognizer-model', include_optimizer=False)
~~~

## 解答を作成する

最後。解答の作成です。

~~~python
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import get_test_data_frame, get_xs


# モデルを取得します。
model = tf.keras.models.load_model('digit-recognizer-model')

# データを取得します。
data_frame = get_test_data_frame()

# 入力データを取得します。
xs = get_xs(data_frame)

# 解答を作成して、保存します。
submission = pd.DataFrame({'ImageId': data_frame.index + 1, 'Label': np.argmax(model.predict(xs, batch_size=256), axis=-1)})
submission.to_csv('submission.csv', index=False)
~~~

やりました！　精度は0.99567で、2,000人位中126位になりました！　カンニング（Digit RecognizerのデータはMNISTという答えが公開されているデータセットをそのまま使用しているので、MNISTを見れば正解が分かってしまう）している連中がいることを考えると、かなり良い成績だと思います。

やったことは、とても簡単ですしね。深層学習と画像認識で検索したら見つかったDenseNetを、Kerasの公式実装を参考にして作成しただけ。こんな簡単なことで高い精度で画像認識できるなんて、深層学習は素晴らしい！

# LightGBMで画像のクラス分類

でも、深層学習は、GPUやTPUがないと遅くてやってられないんですよね……。というわけで、GPUやTPUがない場合向けに、LightGBMでも画像のクラス分類をやってみましょう。

## データセットを取得する

LightGBMの場合のデータセットを取得するモジュールはこんな感じ。

~~~python
import pandas as pd
import os.path as path


# DataFrameを取得します。
def get_data_frame(filename):
    return pd.read_csv(path.join('..', 'input', 'digit-recognizer', filename))


# 訓練用のDataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用のDataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 入力データを取得します。
def get_xs(data_frame):
    return data_frame[list(map(lambda i: f'pixel{i}', range(784)))]


# 正解データを取得します。
def get_ys(data_frame):
    return data_frame['label'].values
~~~

正規化不要で楽ちんですな。LightGBMはやっぱり楽ちんでイイ！

## モデルを保存して読み込む

モデルを保存したり読み込んだりするモジュールはこんな感じ。

~~~python
import lightgbm as lgb
import os.path as path
import pickle

from glob import glob


# LightGBMのパラメーターを保存します。
def save_params(params):
    with open(path.join('digit-recognizer-model', 'params.pickle'), mode='wb') as f:
        pickle.dump(params, f)


# LightGBMのパラメーターを読み込みます。
def load_params():
    with open(path.join('digit-recognizer-model', 'params.pickle'), mode='rb') as f:
        return pickle.load(f)


# モデルを保存します。
def save_model(model):
    for i, booster in enumerate(model.boosters):  # 交差検証なので、複数のモデルが生成されます。
        booster.save_model(path.join('digit-recognizer-model', f'model-{i}.txt'))


# モデルを読み込みます。
def load_model():
    result = lgb.CVBooster()

    for file in sorted(glob(path.join('digit-recognizer-model', 'model-*.txt'))):  # 交差検証なので、複数のモデルが生成されます。
        result.boosters.append(lgb.Booster(model_file=file))

    return result
~~~

例によって切り貼りして置換しただけです。

## ハイパー・パラメーター・チューニング

Optunaに丸投げでハイパー・パラメーター・チューニングをするモジュールはこんな感じ。

~~~python
import optuna.integration.lightgbm as lgb

from dataset import get_train_data_frame, get_xs, get_ys
from model import save_params


# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 10,
    'force_col_wise': True  # LightGBMの警告を除去するために追加しました。
}

# ハイパー・パラメーター・チューニングをします。
tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
cv_result = tuner.run()
model = tuner.get_best_booster()

# LightGBMのパラメーターを保存します。
save_params(tuner.best_params)
~~~

クラス分類なので`objective`を`multiclass`にして、`multiclass`の場合のデフォルトの`metric`である`multi_logloss`を設定しました。あと、クラス分類の場合は何個のクラスに分類するのかという情報が必要なので、`num_class`に10を指定しています。

あとはいつも通りです。が、気を付けていただきたいのですけど、このプログラムは実行にやたらと長時間かかります（私の環境では半日くらいかかりました）。なので、夜間とか、どうしても仕事をしたくない日の勤務時間中とかに実行してみてください。

## 予測モデルを作成

今回は特徴量エンジニアリングの余地がない（画像処理に詳しい人ならできるのかもしれないけど）ので、いきなり予測モデルを作成します。

~~~python
import matplotlib.pyplot as plot
import numpy as np
import optuna.integration.lightgbm as lgb

from dataset import get_train_data_frame, get_xs, get_ys
from model import load_params, save_model


rng = np.random.default_rng(0)

# データを取得します。
data_frame = get_train_data_frame()

# データセットを取得します。
xs = get_xs(data_frame)
ys = get_ys(data_frame)

# LightGBMのパラメーターを取得します。
params = load_params()

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 学習曲線を出力します。
plot.plot(cv_result['multi_logloss-mean'])
plot.show()

# モデルを保存します。
save_model(model)
~~~

完全にいつも通りですね。書くことがなくて困っちゃうくらいにLightGBM簡単ですな。

## 解答を作成する

あとは、解答を作成して、Kaggleに提出するだけ。

~~~python
import numpy as np
import pandas as pd

from dataset import get_test_data_frame, get_xs
from model import load_model


# モデルを読み込みます。
model = load_model()

# データを取得します。
data_frame = get_test_data_frame()
xs = get_xs(data_frame)

# 提出量のCSVを作成します。
submission = pd.DataFrame({'ImageId': data_frame.index + 1, 'Label': np.argmax(np.mean(model.predict(xs), axis=0), axis=-1)})
submission.to_csv('submission.csv', index=False)
~~~

その精度は、0.94128でした。2,000人くらい中の1,740番目くらい。やっぱり画像ならばGPUやTPUを使って深層学習したいけれど、無い袖は振れないのでLightGBMもアリなんじゃないかなぁと。画像の解像度が大きかったりすると、GPUなしで深層学習するより時間がかかったりするかもしれませんけど……。

# Transformerでテーブル・データ


