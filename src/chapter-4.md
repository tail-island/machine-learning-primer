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
            # 画像の該当部分とカーネルの内積を求めて新たな画像を作成します。
            result[y, x] = np.inner(image[y: y + 3, x: x + 3].flatten(), kernel.flatten())

    result[result <   0] =   0  # noqa: E222
    result[result > 255] = 255

    return result


image = cv2.cvtColor(cv2.imread('4.2.07.tiff'), cv2.COLOR_RGB2GRAY)

plot.imshow(image)
plot.show()

plot.imshow(filter(image, np.array([[0, 2, 0], [2, -8, 2], [0, 2, 0]])))
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

＃前の文章の「そんな場合」は遠くに位置する「自然言語のような遠くのものとの関係がある場合」で、ほら、遠くにあるものと関係があるでしょ？

自然言語処理では、単語をベクトルで表現します。「私」＝[0.1, 0.0, 0.4, 0.5, 0.0]みたいな感じ。で、単語のベクトルを要素に持つ行列と単語のベクトルの内積をとる（内積attentionの場合。他に加法attentionってのもありますけどよく知らない）と各単語の重みが出てきて、で、その重みに合わせて単語ベクトルと内積をとって、単語と単語の関係を考慮した行列を作成します。で、これだけだとただの行列計算なので、機械学習できるよう、重みパラメータを追加したのがアテンションです。

あとは、深層学習ってのは本来固定長の入力しか受け付けられないのだけど、それを数式上の工夫で可変長にして、翻訳で精度を大幅に向上させたのがTransformerという深層学習のモデルです。

## 使用するライブラリは、TensorFlow

大雑把な歴史の話が終わりましたので、使用するライブラリを選んでいきましょう。玄人はLuaでTorchかPythonでPyTorchを使うみたいで、深層学習の論文ではこれらの環境が良く使われているみたいです。が、素人の私は敢えてGoogle社のTensorFlowを推薦します。

論文の追試をするとか新しい論文を書くとかなら他の論文と同じ環境が良いのでしょうけど、実際の案件で使用する場合は、やっぱり寄らば大樹の影ですよ。TensorFlowは、様々なプログラミング言語から利用できたり、スマートフォンやIoTデバイスで実行できたりと、周辺機能の充実具合が段違いで優れているので将来のビジネス展開もバッチリですぜ。

## TensorFlowでTransformer

では、いきなりで申し訳ないのですけど、TensorFlowで流行りのTransformerを作ります。

いきなりそんな無茶なと感じた貴方は、[言語理解のためのTransformerモデル](https://www.tensorflow.org/tutorials/text/transformer?hl=ja)を開いてみてください。これ、TensorFlowのチュートリアルの一部なんですけど、ここに懇切丁寧にTransformerの作り方が書いてあります。

チュートリアルというのは、コンピューター業界では手を動かして実際にプログラムを作ることや、入門編という意味で使用されます。TensorFlowのチュートリアルはたぶん両方の意味で、入門レベルの人にも分かる記述で、実際に手を動かしてプログラムを作っている間に嫌でも理解できちゃうという内容になっています。いきなりTransformerは難しそうという場合は、[はじめてのニューラルネットワーク：分類問題の初歩](https://www.tensorflow.org/tutorials/keras/classification?hl=ja)から順を追ってやっていけばオッケーです。

ただね、TensorFlowを作っているような人ってのはとても頭が良い人で、で、頭が良い人ってのはたいていプログラミングが下手なんですよね……。彼らは頭が良いので、複雑なものを複雑なままで理解できます。で、理解した複雑な内容をそのまま複雑なプログラムとして実装しちゃう。ビジネス・アプリケーションのプログラマーが考えているようなリーダビリティへの考慮とかはゼロの、複雑怪奇な糞コードを書きやがるんですよ。

なので、TensorFlowのチュートリアルをやったあとは、できあがったコードをリファクタリングしましょう。その結果は、こんな感じなりました。

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

ただ、シミュレーターを作成可能な場合であっても深層学習が役に立つ場合もあるんですよ。それがどんな場合かというと、シミュレーションにやたらと時間がかかる場合です。実は深層学習はすべての関数を近似できるらしくて（その証明は難しくて私では理解できなかったけど）、ということは、シミュレーションを近似できちゃうというわけ。

たとえば原子の動きのシミュレーションは、量子力学の基本法則に立脚した第一原理計算というのがあって、それを活用する密度汎関数法（density functional theory）という計算でできる（らしい）んです。ただ、これってとにかく計算に時間がかかる（っぽい）んですよ。複雑な原子のシミュレーションはとにかく時間がかかってやってられないので、だからこれを深層学習で近似しちゃおうというのが、Preferred Networks社とENEOS社の[Matlantis](https://matlantis.com/ja/)で、マテリアル・インフォマティクスではとても役に立つ（みたい）です。

こんな感じで、もし遅くてやってられないシミュレーターがあるなら、深層学習で代替できないか検討してみるのも良いと思います。

あ、シミュレーターを作れるかもしれないけど作るのが面倒なのでとりあえず深層学習ってのもアリですよ。私はこっち派です。

# Vision Transformerで画像認識

話を元に戻して、KaggleのGetting Startedの[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)で画像認識をやりましょう。



x

x

x

x

x

### テーブル・データは深層学習「以外」で

ではどうするかというと、テーブル・データで深層学習するときには、昔懐かしい全結合になるんですよね。

アテンションは画像認識等の自然言語以外の分野でも使われていて、今では何をするにもアテンションありきという感じなのですけど、でも、アテンションってベクトルが入力じゃないとダメなんですよ。身長180cmや体重79kg（ちょっとサバ読んだ）のようなデータはスカラー値でベクトルじゃないので、なんとかしてベクトル化しなければなりません（カテゴリーなどの離散量は単語と同じやり方で、連続量は行列を掛け算してバイアスを足すことでベクトル化します）。

ただ、値をなんとかしてベクトル化すればアテンションが使えて、アテンションは遠くのデータとの関係も考慮してくれるから表の一番左のカラムと一番右のカラムの関係も加味した予測をしてくれるはず。

