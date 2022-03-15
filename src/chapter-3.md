# LightGBMで勾配ブースティング

というわけで、まずは勾配ブースティング（gradient boosting）しましょう。使用するライブラリは、LightGBMです。

## 勾配ブースティングとは？

勾配ブースティングは、複数の決定木を使用して予測する機械学習の手法です。……って前にやったランダム・フォレストと同じじゃんと思ったかもしれませんけど、学習のさせ方がランダム・フォレストとは一味違うんです。

### バギングとブースティング

複数の予測器を使う方式はアンサンブル（Ensemble）と呼ぶのですけど、このアンサンブルの代表的な手法に、バギング（bagging）とブースティング（boosting）という2つの手法があります。

そもそもね、まったく同じ思考をする人が3人集まっても同じ答えが3つ返ってくるだけですから、3人いても文殊の知恵は手に入りません。アンサンブルて予測の精度を上げるには、バラツキが必要なんですよ。ではどのようにバラツキを出すのかというと、今やっているのは機械学習なので、データセットを変えることでバラツキがでます。

バギングでは、データセット全体からのサンプリング抽出で複数の異なるデータセットを作成します。サンプリングだからデータセットは異なるはずで、だから学習結果も異なるのでアンサンブルで予測精度が上がるはずという考え方なわけですな。

でも、サンプリング結果って、そんなに違わないですよね？　日本全体からサンプリングで1万人集めた場合、たぶん男女比は日本全体とほぼ同じ。年齢構成だってそう。だから、あるサンプリング結果で学習してもうまく予測できないような問題は、たぶんあまり違わない他のサンプリング結果で学習してもうまく予測できない。

だからサンプリング処理に介入しちゃおうというのがブースティング（boosting）。サンプリングしたデータセットで学習してとりあえず1つの予測器を作成し、その予測器でデータセット全体を予測して、正解したデータは少なめに、不正解だったデータは多めに選択されるように重みづけしてもう一度サンプリングします。その結果は1番目の予測器が苦手とする問題が多く含まれるわけで、だからそのデータセットで作成した2番目の予測器は1番目の予測器が苦手とする予測をうまくこなせるはず。これを繰り返していけば、最終的には、トータルで見れば弱点がない複数の予測器になるというわけ。

こう考えると、ブースティングの精度の高さにうなづけるでしょ？　ただ、予測を間違える方が正しいような外れ値に引っ張られてしまったり、過学習してしまったりする危険性があります。でも精度が高いのでやめられないんですな。

### で、勾配とは？

……えっと、勾配（gradient）は勾配降下法の意味で、調整していくときの方法みたいです。なんかね、偏微分で勾配を求めてってやるらしいんですけど、私レベルでは欠片も理解できてない。

でも大丈夫！　これまでだっていろいろな機械学習の手法でパラメーター調整のためにいろいろ数学の手法が使われていただろうけど、それらをやるのは機械学習ライブラリの役目でプログラマーの私の役割じゃなかったですもんね。勾配降下法でやるんだーというふわっとした理解で大丈夫じゃないかな。

### 勾配ブースティングとは？

というわけで、勾配ブースティングは決定木をアンサンブルして予測する手法で、アンサンブル方法はバギングではなくてブースティングなので精度が高くて、あと、勾配降下法を使うので学習の効率が良い（と思う）な手法で、とにかくお勧めです。

## LightGBMとは？

LightGBMは、マイクロソフト社が開発したオープン・ソースの勾配ブースティングのライブラリです。

勾配ブースティングのライブラリはXGBoostとLightGBMが有名なのですけど、本稿ではLightGBMを使用します。今はあまり差がないみたいですけど、昔はLightGBMの方が圧倒的に速かったので大人気になって、そのまま今も利用者が多くて寄らば大樹の陰なのでLightGBMお勧めです。

# お題は、Kaggleから

使用する道具が決まったのでこれからLightGBMで機械学習していくわけですけど、毎回お題を用意するのは大変すぎて私が死んじゃう。あと、本稿の内容は機械学習の基本だけなので、読後に機械学習を実践してテクニックを学んでいく必要があるのだけど、高度なテクニックを紹介していくのも私では無理。なので、お題の提供と最強テクニックの提供を[Kaggle](https://www.kaggle.com/)に丸投げしちゃいましょう。

Kaggle（カグルと読みます）は、機械学習のコンペのサイトです。企業や政府が課題と賞金を出して、Kaggler（カグラーと読みます。Kaggleの参加者の意味です）が競争形式でその課題を解いていきます。

競争なんてやりたくない、私はひっそり機械学習を学んでアルゴリズム作成で楽をしたいだけなんだ……という方にもKaggleは役に立ちます（実は、私は怖くてKaggleのコンペはやっていません）。Kaggleには練習用のコンペが用意されていますし、Codeに他の参加者が実際に動くコードで解説を投稿してくれますし、今競技中で戦っている最中なのに面白いことを思いついたKagglerがDiscussionにそのアイデアを書いてくれちゃいます。実践環境を提供してくれる上に、異常に能力が高い方々のアドバイスが盛りだくさんなのですから、機械学習の勉強が捗りまくります。

……Kaggleは全部英語だけどな。でも大丈夫、英語ダメダメな私だけど、Chrome翻訳と[DeepL](https://www.deepl.com/ja/translator)が私を助けてくれるはず！

# 勾配ブースティングで2値分類

というわけで、Kaggleからお題をもらってLightGBMで勾配ブースティングでクラス分類（2値分類ですが）をやりましょう。お題はKaggleが最初の挑戦としてお勧めしている[Titanic](https://www.kaggle.com/c/titanic)にします。Titanicは、Kaggleのページの左側の\[Competitions\]をクリックして、\[Getting Started\]をクリックするとリスト表示される中に入っています。

というわけで、Titanicを開いてデータをダウンロードして、本稿のソース・コードのtitanic/input/titanicに保存してください。データはtrain.csvとtest.csvに分かれていて、train.csvを使って機械学習して、test.csvを使って予測した結果で精度を競うわけですな。

## まずは、予測可能なのか調べる

さて、今回はKaggleなので他の参加者のスコアを見れば予測が可能でそこそこの精度が出ることは一目瞭然なのですけど、実際の案件ではそもそも予測が可能なのか分からないことが稀によくあります。

予測ができない理由としては、予測する内容とは無関係なデータしかない場合と、そもそも予測対象のランダム性が高すぎて予測が不可能な場合等があります。特に結果のランダム性が高い場合は辛くて、たとえば過去10回のサイコロの出目がこんな場合に次に出る目を予測するなんてのは、絶対に無理でしょ？

で、いろいろデータを解析していけば予測が可能か分かるのでしょうけど、私はプログラマーなのでとりあえずプログラムを組んで調べています。こんな感じ。

~~~python
import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from funcy import count
from sklearn.metrics import accuracy_score


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked')))  # factorize()で数値に変換することもできるのですけど、その方式は、実際に予測するときに使えない。。。


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')  # astype('category')しておけば、LightGBMがカテゴリ型として扱ってくれて便利です。

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# データを読み込んで、前準備をします。
data_frame = pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv'))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データセットを取得します。
train_xs = xs[200:]
train_ys = ys[200:]

# 検証データセットを取得します。test.csvを使ってKaggleに問い合わせる方式は、面倒な上に数をこなせないためです。
valid_xs = xs[:200]
valid_ys = ys[:200]

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'binary',  # 2値分類。
    'force_col_wise': True  # 警告を消すために付けました。
}

# 交差検証で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 各特徴量の重要性を出力します。
for booster in model.boosters:
    print(pd.DataFrame({'feature': booster.feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# 精度を出力します。
print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
~~~

前にやったバイクのジャンル予測ではNaNが無くなるように工夫したりカテゴリー値をone hot encodingしたりといろいろ工夫しましたけど、LightGBMはどちらの作業もやらねくて大丈夫とても楽ちんです（できればNaNは埋めた方が良いけど）。ほら、とりあえずLightGBMって気持ちになるでしょ？

残る前準備のカテゴリーの数値への変換は必要ですけど、pandasの`factorize()`と`map()`を使えば、上のコードの`get_categorical_features()`や`get_xs()`のように簡単に書けます。LightGBMがカテゴリーかどうか判断できるよう、カテゴリーのカラムでは`ascategory('category')`しておくのを忘れないようにしてください。

データセットができたら、いつも通りに訓練データセットと検証データセット（後述する交差検証をしているので、テスト・データに近いけど）に分けて、機械学習していきます。

LightGBMでは、どのように機械学習を進めるのかをdict型の変数で指定します。上のコードでの`params`がそれ。とりあえずは、`objective`で何をしたいのかを指定しておけばオッケーです。詳細はLightGBMのドキュメントの[Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)に書いてありますので機械学習に慣れた頃に読んでみてください。今回は生き残ったかどうかの2値分類なので、`objective`に`binary`を指定しました。

で、一般的にはLightGBMは`train()`メソッドで機械学習するのですけど、上のコードでは`cv()`メソッドで学習をしています。`cv()`はcross validationの略で、日本語で交差検証と呼びます。交差検証は過学習していないかを確認するための方法の一つで、これまでにやってきた訓練データと検証データを分割する（これをホールド・アウト法と呼びます）の仲間、データをいくつかに分割して、そのうちの一つを検証データにして残りを訓練データにするってのを、分割したデータすべてでやって精度を出し、結果を平均化して過学習していないか判断します。

![交差検証]()

過学習していないかの検証の方法は学習とは無関係に思えるかもしれませんけど、過学習したら学習を止めるという意味で使えるんです。ただ、検証データで精度が高かったからどんなデータでも高い精度を出せるかというとそうでもなくて、たまたま検証データの癖に合致したので精度が高かったという危険性があります。特に、今回のようにデータ数が少ない場合はその可能性が高い。そんなあやふやな数値に頼るのは危険ですから、交差検証しながら学習する`cv()`メソッドを使用するというわけ。

あとは、のちの作業のために予測にどのカラムがどの程度使われたのかと精度を出力して、学習がどのように進んだのかを示す学習曲線を出力します。

![学習曲線]()

学習曲線で表示しているのは正解と予測の誤差の推移で、縦軸は`object`は`binary`の場合の`metric`のデフォルトである`binary_logloss`（交差検証しているのでその平均）です。これが小さいほど精度が高くなるというわけ。この図の横軸は何かというと、機械学習の用語でエポック（epoch）と呼ばれるもので、データセットを上から下まで全部使用すると1エポックになります。2エポック目では、同じデータセットをやっぱり上から下まで全部使用して学習します。まさに同じ問題集を繰り返し学習しているわけで、だからその問題集に特化した予測をするようになって、過学習して他の問題集の問題を解けなくなっていく可能性が高い。

で、学習曲線を見てみると、20エポックのあたりから精度がだんだん悪くなっているので、まさに過学習していることが分かります。でも、エポック数を減らしてもう一度機械学習しなくても大丈夫です。LightGBMは、デフォルトだと最も精度が高かったパラメーターを使用して予測をするようになっているので、先のプログラムを実行すると表示される精度の0.795は、20回目あたりのパラメーターを使用した場合の値というわけですな。

ともあれ、8割近く正解できているのであれば、うん、Titanicは予測可能な問題ですね。これで安心できたので、精度向上のための施策をやることにしましょう。

## 特徴量エンジニアリングで精度を上げる

前に、LightGBMは勾配ブースティングで、勾配ブースティングは決定木をバギングするものだと述べました。その決定木ってのは、たしか`if/else`で予測する仕組みでしたよね？　コードにすると、こんな感じでした。

~~~python
if x <= a1:
    if x <= a2:
        return b1
    else
        return b2
else
    if x <= a3:
        return b3
    else
        return b4
~~~

このコードを眺めていると、たとえば、給料が少ないのでこっそり副業している場合に新しいバイクを買うか予測するような場合に効率が悪いことが分かります。データのカラムが給料と副業の収入に分かれていると、給料がいくら以上で、副業の収入がいくら以上だったら買えるみたいな多段の`if`にせざるを得なくて、しかも、給料10万円副業100万円や給料100万円副業10万円みたいな様々な組み合わせの`if`文を作らなければなりません。

ではどうすればよいかというと、給料＋副業の収入を表現する総収入というカラムを追加してあげればよいわけ。Titanicのデータだと、SibSp（同乗した兄弟姉妹と配偶者の数）とParch（同乗した両親と子供の数）がまさにこのケースになります。同乗者がいないと助けてくれる人がいないので死にやすそうですし、同乗者が多いと行動が遅くなるのでやっぱり死にやすそう。だから同乗者の数を入力に追加してあげると精度が向上するんじゃないかな。

このようにカラム（機械学習の人は特徴量（feature）と呼びます）を追加したり、取捨選択したりすることを特徴量エンジニアリングと呼びます。効率が悪かったり無関係だったりするデータから予測をするのは難しいので、機械学習の気持ちになって、できるだけ気持ちよく予測できるようにしてあげる作業です。

で、一般には関係がありそうなデータをかき集めてくる作業になるのですけど、Kaggleだとデータが決められているので、既存の特徴量から新たな特徴量を作成する作業になります。

他の特徴量も見ていきましょう。前回は使わなかったNameの活用を考えてみます。NameにはMr.とかMrs.とかの肩書が含まれていて、肩書から性別や年齢が推測できそうな気がします（Titanicの特徴量には性別や年齢もあるけど、年齢はNaNの場合があるしね）。男性よりも女性、大人よりも子供の方が優先して避難させてもらえそうな気がするので、肩書を追加しましょう。あとは、Fareってのはどうもチケットの合計金額っぽくて、チケットを複数枚買っている場合は高くなっているみたい。チケット料金が安い船室の人は避難が後回しになって死んじゃう可能性が高そうな気がしますから、Fareを家族数で割って単価を出しておきましょう。

というわけで、特徴量エンジニアリングした結果はこんな感じ。

~~~python
import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat
from sklearn.metrics import accuracy_score


# 特徴量を追加します。
def add_features(data_frame):
    # train.csvでの肩書の内訳は、以下の通り。
    # Mr.        509
    # Miss.      180
    # Mrs.       125
    # Master.     40  少年もしくは青年への敬称らしい
    # Dr.         11
    # Col.        10
    # Rev.         6  聖職者への敬称らしい
    # Don.         2
    # Major.       2
    # Mme.         1
    # Ms.          1
    # Capt.        1
    # NaN.         3

    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    # 家族の人数を追加します。
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    # 料金は合計みたいなので、単価を追加します。
    data_frame['FareUnitPrice'] = data_frame['Fare'] / data_frame['FamilySize']

    return data_frame


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データセットを取得します。
train_xs = xs[200:]
train_ys = ys[200:]

# 検証データセットを取得します。test.csvを使ってKaggleに問い合わせる方式は、面倒な上に数をこなせないためです。
valid_xs = xs[:200]
valid_ys = ys[:200]

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'binary',  # 2値分類。
    'force_col_wise': True  # 警告を消すために付けました。
}

# 交差検証で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 各特徴量の重要性を出力します。
for booster in model.boosters:
    print(pd.DataFrame({'feature': booster.feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# 精度を出力します。
print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
~~~

前のコードに、特徴量を追加する処理を追加しただけですね。精度が0.795→0.81に上がりましたので、特徴量エンジニアリングの効果を確認できました。

## 統計の手法を使えば特徴量エンジニアリングは効率化できる

先ほどの特徴量エンジニアリングなのですが、実は統計の手法を使うと効率よく作業できる（みたい）です。

仮説を立てて仮説をプログラミングして精度を検証する、というのが前で述べたやり方なのですけど、仮説を立てる際にもプログラミングの前に仮説の妥当性を検証するにも、統計の手法が有効（っぽい）んです。

でも私は文系で統計手法を知らないので、勢いだけで仮説を立てて速攻でプログラミングして、検証してみたら精度が上がらなくて木端微塵にされるってのを繰り返しています。皆様は私を反面教師に統計手法を勉強（pandasでの統計量の可視化手法のマスターでも可）するのが良いかと愚考します。

私はこれからも「統計で調べるより短い時間で仮説をプログラミングすれば問題はないんだよ！」と言い張るるもりだけどな！

## Optunaでハイパー・パラメーター・チューニング

LightGBMのハイパー・パラメーターはドキュメントの[Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)に書いてあって、チューニングのやり方は[Parameter Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)に書いてあるのですけど、読むのが面倒くさい……。

でも精度向上のためにハイパー・パラメーター・チューニングをやりたい！　というわけで、ツールに頼りましょう。PFNが[Optuna](https://github.com/optuna/optuna)というオープン・ソースのツールを提供してくださっていて、これを使えば全自動でハイパー・パラメーターをチューニングできるんです。

Optunaを使うコードはこんな感じ。

~~~python
import optuna.integration.lightgbm as lgb
import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat


# 特徴量を追加します。
def add_features(data_frame):
    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    # 家族の人数を追加します。
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    # 料金は合計みたいなので、単価を追加します。
    data_frame['FareUnitPrice'] = data_frame['Fare'] / data_frame['FamilySize']

    return data_frame


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。Optunaを信用しているので検証は不要と考え、検証データは作成しません。データ量が多い方が正確なハイパー・パラメーターになりますし。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'binary',  # 2値分類。
    'force_col_wise': True  # 警告を消すために付けました。
}

# 交差検証でハイパー・パラメーター・チューニングをします。
tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
cv_result = tuner.run()
model = tuner.get_best_booster()

# 各特徴量の重要性を出力します。
for booster in model.boosters:
    print(pd.DataFrame({'feature': booster.feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# ハイパー・パラメーターを出力します。
print(tuner.best_params)
~~~

一目見て分かるように、ほとんどコードは変わりません。`cv()`じゃなくて`train()`する場合であれば、`import lightgbm as lgb`を`import optuna.integration.lightgbm as lgb`に変えるだけという楽ちん仕様です。`cv()`の場合でも、上のコードを参考に`LightGBMTunerCV`を使うように修正するだけでオッケー。

たったこれだけで、ハイパー・パラメーター・チューニングは終了です。このプログラムの実行にはちょっと時間がかかりますけど、手でハイパー・パラメーター・チューニングするより速いし、なにより楽ちんです。

## 特徴量エンジニアリングとハイパー・パラメーター・チューニングの成果を確認する

これまでの成果を確認してみましょう。2つ前の特徴量エンジニアリングのコードに、Optunaが作成したパラメーターを埋め込みます。

~~~python
import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat
from sklearn.metrics import accuracy_score


# 特徴量を追加します。
def add_features(data_frame):
    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    # 家族の人数を追加します。
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    # 料金は合計みたいなので、単価を追加します。
    data_frame['FareUnitPrice'] = data_frame['Fare'] / data_frame['FamilySize']

    return data_frame


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データセットを取得します。
train_xs = xs[200:]
train_ys = ys[200:]

# 検証データセットを取得します。test.csvを使ってKaggleに問い合わせる方式は、面倒な上に数をこなせないためです。
valid_xs = xs[:200]
valid_ys = ys[:200]

# LightGBMのパラメーターを作成します。Optunaが作成したパラメーターを使用します。
params = {
    'objective': 'binary',
    'force_col_wise': True,
    'feature_pre_filter': False,
    'lambda_l1': 1.4361833756015463,
    'lambda_l2': 2.760985217750726e-07,
    'num_leaves': 5,
    'feature_fraction': 0.4,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20
}

# 交差検証で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 各特徴量の重要性を出力します。
for booster in model.boosters:
    print(pd.DataFrame({'feature': booster.feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False))

# 精度を出力します。
print(f'Accuracy = {accuracy_score(valid_ys, np.mean(model.predict(valid_xs), axis=0) >= 0.5)}')

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
~~~

精度は、0.81→0.82になりました。　まぁ精度は誤差の気もしますけど、学習曲線がとてもきれいになっていてとても嬉しい！

![学習曲線]()

過学習を防ぐ方法は、よりシンプルな機械学習モデルを作るか、データ量を増やすかになります（パラメーターを正則化するというのもありますけど）。で、Optunaが絶妙なシンプル具合になるハイパー・パラメーターを作成してくれたので、過学習しない学習曲線になったというわけです。

さて、ここまで話がとんとん拍子に進んだように書いてきましたけど、実際は、こんなにうまくは進みません。特徴量エンジニアリングで精度が上がっても、ハイパー・パラメーター・チューニングで特徴量の扱われ方が変わって効果が微妙になったりね。なので実際は、とりあえず特徴量エンジニアリング→Optunaでハイパー・パラメーター・チューニング→チューニングされたハイパー・パラメーターを使用してひたすら特徴量エンジニアリング（ときどきはハイパー・パラメーター・チューニングもやり直す）という感じになります。

で、そのひたすら繰り返す特徴量エンジニアリングで役に立つのが、特徴量の重要性です。プログラムを実行すると、以下のような情報が出力されていましたよね？

~~~
         feature  importance
2            Age        90.0
9  FareUnitPrice        64.8
0         Pclass        61.0
5           Fare        44.4
6       Embarked        34.4
8     FamilySize        32.8
7          Title        25.4
1            Sex        17.2
4          Parch        13.4
3          SibSp         7.4
~~~

これ、特徴量が決定木で使われた数です。おお、やっぱり若者と金持ちが優先で、だからおっさんで貧乏な私は後回しなんだと世の中が理解できるようになって、特徴量エンジニアリングが進むんじゃないかな。

## 予測モデルを作成する

さて、これでチューニングは概ね終了したので、結果をKaggleに提出して採点したい。Kaggleに提出するだけなら学習から予測まで一気通貫で実施するプログラムで良いのですけど、実際にシステムを開発するときはそんなわけにはいきません。本番環境で学習するなんて無駄ですもんね。学習は事前にやっておけばよいんです。

だから、まずは、機械学習して予測モデルを作成するプログラムです。こんな感じ。

~~~python
import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import pickle
import os
import os.path as path

from functools import reduce
from funcy import count, repeat


# 特徴量を追加します。
def add_features(data_frame):
    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    # 家族の人数を追加します。
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    # 料金は合計みたいなので、単価を追加します。
    data_frame['FareUnitPrice'] = data_frame['Fare'] / data_frame['FamilySize']

    return data_frame


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# 正解を取得します。
def get_ys(data_frame):
    return data_frame['Survived']


# 機械学習モデルを保存します。
def save_model(model, name):
    for i, booster in enumerate(model.boosters):  # 交差検証なので、複数のモデルが生成されます。
        booster.save_model(path.join('titanic-model', f'{name}-{i}.txt'))


# カテゴリ型の特徴量を、どの数値に変換するかのdictを保存します。
def save_categorical_features(categorical_features):
    with open(path.join('titanic-model', 'categorical-features.pickle'), mode='wb') as f:
        pickle.dump(categorical_features, f)


# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'train.csv')))
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。できるだけ精度を上げたいので、すべてのデータを使用して機械学習します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。Optunaが作成したパラメーター（+ learning_rate）を使用します。
params = {
    'objective': 'binary',
    'force_col_wise': True,
    'feature_pre_filter': False,
    'lambda_l1': 1.4361833756015463,
    'lambda_l2': 2.760985217750726e-07,
    'num_leaves': 5,
    'feature_fraction': 0.4,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'learning_rate': 0.01
}

# 交差検証で機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), num_boost_round=1000, return_cvbooster=True)
model = cv_result['cvbooster']

# モデル保存用のディレクトリを作成します。
os.makedirs('titanic-model', exist_ok=True)

# モデルを保存します。
save_model(model, 'model')
save_categorical_features(categorical_features)

# 学習曲線を出力します。
plot.plot(cv_result['binary_logloss-mean'])
plot.show()
~~~

LightGBMのパラメーターに`learning_rate`を追加しているところに注意してください。勾配降下法では正解との誤差に基づいてパラメーターを調整していくのですけど、`learning_rate`を設定することで、大きく一気に調整したり小さく小刻みに調整したりできます。`learning_rate`が大きければ学習が早く終わりますけど調整が大雑把なので精度はそんなに高くない、小さい場合はその逆で時間はかかるけど精度が高くなります。

特徴量エンジニアリングするときには早くサイクルを回したいので`learning_rate`を大きく（デフォルト値でも十分に大きい）、最終成果物を作るときには`learning_rate`を小さくします。今回は、デフォルト値の1/10に設定しました。

で、`learning_rate`を下げた分だけ学習が遅くなりますから、エポック数に相当する`num_boost_round`（`cv()`の引数です）の値を大きくしています。

あとは、交差検証なので複数のモデルが作られるのでそれらを全部保存することと、カテゴリーの特徴量を数値化するための情報も帆損しなければなりません。それ以外は、少しでも精度を上げるために検証データを作らずにすべてのデータで学習しているくらいかな。

あ、今回はKaggleのnotebookで実行しやすいようにモジュール化していませんけど、モジュール化すればもっとシンプルなコードにできます。実案件ではモジュール化してみてください。

## 予測モデルで予測する

実際のシステムでは本番環境で動く部分に相当する、予測をするプログラムを作ります。保存された予測モデルをロードして、test.csvを使用して予測をしてKaggle提出量のCSVファイルを作成します。

~~~python
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os.path as path

from functools import reduce
from funcy import count, repeat
from glob import glob


# 特徴量を追加します。
def add_features(data_frame):
    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    # 家族の人数を追加します。
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']

    # 料金は合計みたいなので、単価を追加します。
    data_frame['FareUnitPrice'] = data_frame['Fare'] / data_frame['FamilySize']

    return data_frame


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))), ('Sex', 'Embarked', 'Title')))


# データを取得します。
def get_xs(data_frame, categorical_features):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        # data_frame[feature] = data_frame[feature].map(mapping | {np.nan: -1}).astype('category')  # KaggleのNotebookのPythonのバージョンが古くて、merge operatorが使えなかった。
        data_frame[feature] = data_frame[feature].map({**mapping, **{np.nan: -1}}).astype('category')

    # 予測に使用するカラムだけを抽出します。NameとTicketは関係なさそうなので無視、Cabinは欠損地が多いので無視しました。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'FareUnitPrice']]


# Kaggleのnotebookなのかを判定します。
def is_kaggle_notebook():
    return '_dh' in globals() and globals()['_dh'] == ['/kaggle/working']


# モデルをロードします。
def load_model(name):
    result = lgb.CVBooster()

    base_path = '.' if not is_kaggle_notebook() else path.join('..', 'input')  # KaggleのnotebookのDatasetは../inputに展開されます。。。

    for file in sorted(glob(path.join(base_path, 'titanic-model', f'{name}-*.txt'))):  # 交差検証なので、複数のモデルが生成されます。
        result.boosters.append(lgb.Booster(model_file=file))

    return result


# カテゴリ型の特徴量を、どの数値に変換するかのdictをロードします。
def load_categorical_features():
    base_path = '.' if not is_kaggle_notebook() else path.join('..', 'input')  # KaggleのnotebookのDatasetは../inputに展開されます。。。

    with open(path.join(base_path, 'titanic-model', 'categorical-features.pickle'), mode='rb') as f:
        return pickle.load(f)


# モデルをロードします。
model = load_model('model')
categorical_features = load_categorical_features()

# データを読み込んで、前準備をします。
data_frame = add_features(pd.read_csv(path.join('..', 'input', 'titanic', 'test.csv')))
data_frame['Fare'] = data_frame['Fare'].fillna(data_frame['Fare'].mean())  # train.csvにはないけど、test.csvのFareにはNaNがある。。。

# 予測用のデータを取得します。
xs = get_xs(data_frame, categorical_features)

# 予測して、結果をCSVとして保存します。
submission = pd.DataFrame({'PassengerId': data_frame['PassengerId'], 'Survived': (np.mean(model.predict(xs), axis=0) >= 0.5).astype(np.int32)})
submission.to_csv('submission.csv', index=False)
~~~

機械学習モデルの読み込みはLightGBMの`Booseter`のコンストラクタで`model_file`を指定すればオッケー。交差検証なので複数のモデルがあることだけ注意。あと、Kaggleのnotebookで実行する場合、追加したファイルは../inputに展開されるので、Kaggleのnotebookの場合用の分岐も入れています。残りの予測をする部分は、これまでと同じです。

これでプログラミングがすべて完了したので、作成されたCSVファイルをKaggleに提出して、スコアを見てみると……0.77990で3,282位でした。スコアが0.8くらいの人のやり方を参考にすれば、もう少し精度を上げられるかもしれません。Leaderboardに並んでいるスコア1.0の連中は[カンニングしている](https://www.kaggle.com/maryragozina/notebook9849f51564?scriptVersionId=85357609)ので、参考にしては駄目です。

あとね、いくらスコアが高くても、たまたまtest.csvに合っていただけという可能性もあるんですよ……。Kaggleのコンペでは競技中はテスト・データの一部を使用したスコアでの順位しか表示されず、最終的な順位はテスト・データの残りでスコアで決定されます。テスト・データの一部に過剰にフィットしていただけだったので、競技中はスコアが高かったけど最終的には低かったなんて場合もあります。だから、スコアを上げるためだけにさらにハイパー・パラメーターを手でチューニングしたりするのは無意味だと思います。

というかね、0.77990って、スコアの最頻値よりも良い、そこそこ自慢して良いスコアだと思うんですよ（精度なので、大きいほど良いです）。公式のチュートリアルのスコアが0.77751みたいですし、実際の案件だったらそれでビジネスが成り立つ精度であれば十分なわけですし。

![Titanicのスコアのヒストグラム](./image/score-histgram-titanic.png)

というわけで、LightGBMとOptunaを使って、あと少しだけ特長量エンジニアリングすれば、機械学習で簡単にそこそこの精度を出せることがわかりました。しかも、特長量エンジニアリングまで含めて、全てが慣れ親しんだプログラミング作業です。

ほら、アルゴリズムを考えるのが面倒だからとりあえず機械学習してみるってのは、十分アリだと思いませんか？

# 勾配ブースティングで回帰

続いて、やはりKaggleのGetting Startedの[House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)をやってみましょう。今度は回帰分析です。あと、Titanicではやらなかったモジュール化でやります。KaggleのNotebookで動かすのは少し大変かもしれませんけど気にしない方向で。

## データセット

データセットを取得するモジュールを作成します。

~~~python
import pandas as pd
import os.path as path

from funcy import concat, count, repeat


# 大小関係がある文字列の特長量を数値に変換します。
def convert_to_number(data_frame):
    for feature, names in concat(zip(('Utilities',),
                                     repeat(('AllPub', 'NoSewr', 'NoSeWa', 'ELO'))),
                                 zip(('LandSlope',),
                                     repeat(('Gtl', 'Mod', 'Sev'))),
                                 zip(('ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'),
                                     repeat(('Ex', 'Gd', 'Ta', 'Fa', 'Po'))),
                                 zip(('BsmtExposure',),
                                     repeat(('Gd', 'Av', 'Mn', 'No'))),
                                 zip(('BsmtFinType1', 'BsmtFinType2'),
                                     repeat(('GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'))),
                                 zip(('Functional',),
                                     repeat(('Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'))),
                                 zip(('GarageFinish',),
                                     repeat(('Fin', 'RFn', 'Unf'))),
                                 zip(('Fence',),
                                     repeat(('GdPrv', 'MnPrv', 'GdWo', 'MnWw')))):
        data_frame[feature] = data_frame[feature].map(dict(zip(names, count(len(names) - 1, -1)))).fillna(-1).astype('int')

    return data_frame


# 特徴量エンジニアリングで、特徴量を追加します。
def add_features(data_frame):
    data_frame['TotalSF'] = data_frame['TotalBsmtSF'] + data_frame['1stFlrSF'] + data_frame['2ndFlrSF']  # 3階建てはない？
    data_frame['SFPerRoom'] = data_frame['TotalSF'] / data_frame['TotRmsAbvGrd']

    return data_frame


# 訓練用のDataFrameを読み込みます。
def get_train_data_frame():
    return add_features(convert_to_number(pd.read_csv(path.join('..', 'input', 'house-prices-advanced-regression-techniques', 'train.csv'))))


# テスト用のDataFrameを読み込みます。
def get_test_data_frame():
    return add_features(convert_to_number(pd.read_csv(path.join('..', 'input', 'house-prices-advanced-regression-techniques', 'test.csv'))))


# カテゴリーの特長量を変換します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))),
                    ('MSZoning',
                     'Street',
                     'Alley',
                     'LotShape',
                     'LandContour',
                     'LotConfig',
                     'Neighborhood',
                     'Condition1',
                     'Condition2',
                     'BldgType',
                     'HouseStyle',
                     'RoofStyle',
                     'RoofMatl',
                     'Exterior1st',
                     'Exterior2nd',
                     'MasVnrType',
                     'Foundation',
                     'Heating',
                     'CentralAir',
                     'Electrical',
                     'GarageType',
                     'PavedDrive',
                     'MiscFeature',
                     'SaleType',
                     'SaleCondition')))


# 入力データを読み込みます。
def get_xs(data_frame, categorical_features):
    for feature, mapping in categorical_features.items():
        data_frame[feature] = data_frame[feature].map(mapping).fillna(-1).astype('category')

    return data_frame[['MSSubClass',
                       'MSZoning',
                       'LotFrontage',
                       'LotArea',
                       'Street',
                       'Alley',
                       'LotShape',
                       'LandContour',
                       'Utilities',
                       'LotConfig',
                       'LandSlope',
                       'Neighborhood',
                       'Condition1',
                       'Condition2',
                       'BldgType',
                       'HouseStyle',
                       'OverallQual',
                       'OverallCond',
                       'YearBuilt',
                       'YearRemodAdd',
                       'RoofStyle',
                       'RoofMatl',
                       'Exterior1st',
                       'Exterior2nd',
                       'MasVnrType',
                       'MasVnrArea',
                       'ExterQual',
                       'ExterCond',
                       'Foundation',
                       'BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinSF1',
                       'BsmtFinType2',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'Heating',
                       'HeatingQC',
                       'CentralAir',
                       'Electrical',
                       '1stFlrSF',
                       '2ndFlrSF',
                       'LowQualFinSF',
                       'GrLivArea',
                       'BsmtFullBath',
                       'BsmtHalfBath',
                       'FullBath',
                       'HalfBath',
                       'BedroomAbvGr',
                       'KitchenAbvGr',
                       'KitchenQual',
                       'TotRmsAbvGrd',
                       'Functional',
                       'Fireplaces',
                       'FireplaceQu',
                       'GarageType',
                       'GarageYrBlt',
                       'GarageFinish',
                       'GarageCars',
                       'GarageArea',
                       'GarageQual',
                       'GarageCond',
                       'PavedDrive',
                       'WoodDeckSF',
                       'OpenPorchSF',
                       'EnclosedPorch',
                       '3SsnPorch',
                       'ScreenPorch',
                       'PoolArea',
                       'PoolQC',
                       'Fence',
                       'MiscFeature',
                       'MiscVal',
                       'MoSold',
                       'YrSold',
                       'SaleType',
                       'SaleCondition',
                       'TotalSF',
                       'SFPerRoom']]


# 正解データを読み込みます。
def get_ys(data_frame):
    return data_frame['SalePrice']
~~~

House Pricesは特徴量の数が多くて大変でした……。なんでGetting Startedにこんなに特徴量が多い問題があるのかというと、統計的な数値演算で予測する場合向けに、予測に役立つ特徴量を取捨選択する手法を学ぶためだと思うんですよ。でも、我々は勾配ブースティングで決定木なので、取捨選択は機械学習が勝手にやってくれます。というわけで、全部の特徴量を使用してやりました。

で、日本人ならば5段階評価にするところを、なんでか欧米人は文字で表現する場合があるんですよね……。たとえば、Excellent、Good、Average、Typical、Fair、Poorとか。これを数値に変換する`convert_to_number`関数を作成しました。Funcyを使うと関数型プログラミングできるので楽ちんです（関数型プログラミングに慣れていないと読みづらいかもしれませんけど……）。で、データを見たところ最低点未満の場合にNaNになっているみたいだったので、NaNを最低値の-1で埋める処理もしています。

あと、上のコードでは特長量エンジニアリングの結果の`TotalSF`（総床面積）と`SFPerRoom`（部屋単位の床面積）を追加されていますけど、これは作成後に特徴量エンジニアリングして精度を検証してを繰り返した際に作成したもので、このコードを最初に作成したときにはもちろんなかったことに注意してください。

他は、特に工夫なしです。Titanicの時のコードを切り貼りして作成しました。

## モデルの保存と読み込み

今回はローカルで作業していますので、作業中に出来た成果物をストレージに保存したり読み込んだりできます。そこで、モデルの保存と読み込み用のモジュールも作成しました。

~~~python
import lightgbm as lgb
import os.path as path
import pickle

from glob import glob


# LightGBMのパラメーターを保存します。
def save_params(params):
    with open(path.join('house-prices-model', 'params.pickle'), mode='wb') as f:
        pickle.dump(params, f)


# LightGBMのパラメーターを読み込みます。
def load_params():
    with open(path.join('house-prices-model', 'params.pickle'), mode='rb') as f:
        return pickle.load(f)


# モデルを保存します。
def save_model(model):
    for i, booster in enumerate(model.boosters):  # 交差検証なので、複数のモデルが生成されます。
        booster.save_model(path.join('house-prices-model', f'model-{i}.txt'))


# モデルを読み込みます。
def load_model():
    result = lgb.CVBooster()

    for file in sorted(glob(path.join('house-prices-model', 'model-*.txt'))):  # 交差検証なので、複数のモデルが生成されます。
        result.boosters.append(lgb.Booster(model_file=file))

    return result


# カテゴリーの特徴量を保存します。
def save_categorical_features(categorical_features):
    with open(path.join('house-prices-model', 'categorical-features.pickle'), mode='wb') as f:
        pickle.dump(categorical_features, f)


# カテゴリーの特徴量を読み込みます。
def load_categorical_features():
    with open(path.join('house-prices-model', 'categorical-features.pickle'), mode='rb') as f:
        return pickle.load(f)
~~~

Titanicの時との違いは、LightGBMのパラメーターもストレージに保存するようにしたことです。ターミナルからコピーしてコードにペーストするのは面倒でしたもんね。

あとはTitanicの時と同じ。切り貼りと置換で速攻作成しました。

## ハイパー・パラメーター・チューニング

今回は、特徴量エンジニアリングする前にハイパー・パラメーター・チューニングをしました（データセットのモジュールが正しいかを確認するために、最初に一度手作りのハイパー・パラメーターで精度の確認をしたけど）。

私は特徴量エンジニアリングを直観だけでやっていて、その際に役立つのがLightGBMが出力してれる重要な特徴量で、それはハイパー・パラメーターの値によって結構変わってくるんですよ。だから特徴量エンジニアリングの前にハイパー・パラメーター・チューニングで、例によってOptunaに丸投げで楽しています。

~~~python
import optuna.integration.lightgbm as lgb
import numpy as np
import pandas as pd

from dataset import get_train_data_frame, get_categorical_features, get_xs, get_ys
from model import save_params


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを作成します。
params = {
    'objective': 'regression',
    'metric': 'l2'  # Optunaでエラーが出たので、regressionの場合のデフォルトのmetricを設定しました。
}

# ハイパー・パラメーター・チューニングをします。
tuner = lgb.LightGBMTunerCV(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, optuna_seed=0)
cv_result = tuner.run()
model = tuner.get_best_booster()

# 重要な特徴量を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

# LightGBMのパラメーターを保存します。
save_params(tuner.best_params)
~~~

今回は回帰なので、`objective`に`regression`を指定しています。LightGBMで機械学習するだけならこれだけでも大丈夫なのですけど、Optunaでエラーが出たので`metric`に`regression`の場合のデフォルト値である`l2`を指定しました。

## 精度を確認

特徴量エンジニアリングは、特徴量を追加して精度を確認するという作業を繰り返さなければなりませんので、精度を確認するためのモジュールも作成します。

~~~python
import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from dataset import get_train_data_frame, get_categorical_features, get_xs, get_ys
from model import load_params
from sklearn.metrics import mean_squared_error


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# 訓練データセットを取得します。
train_xs = xs[400:]
train_ys = ys[400:]

# 検証データセットを取得します。
valid_xs = xs[:400]
valid_ys = ys[:400]

# LightGBMのパラメーターを取得します。
params = load_params()

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(train_xs, label=train_ys), return_cvbooster=True)
model = cv_result['cvbooster']

# 重要な特徴量を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

# スコアを出力します。
print(f'Score = {np.sqrt(mean_squared_error(np.log(valid_ys), np.log(np.mean(model.predict(valid_xs), axis=0))))}')

# 学習曲線を出力します。
plot.plot(cv_result['l2-mean'])
plot.show()
~~~

特徴量エンジニアリングでデータセットのモジュールを修正したら、このモジュールを実行するわけですね。で、ずっと特徴量エンジニアリングしていると疲れちゃうので、そんな時にはハイパー・パラメーター・チューニングのモジュールを実行してOptunaが新しいハイパー・パラメーターを作成してくれるまで休憩する感じです。

## モデル作成

特徴量エンジニアリングが完了したら、モデルを作成します。

~~~python
import lightgbm as lgb
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from dataset import get_train_data_frame, get_categorical_features, get_xs, get_ys
from model import load_params, save_model, save_categorical_features


# データを取得します。
data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)

# データセットを取得します。
xs = get_xs(data_frame, categorical_features)
ys = get_ys(data_frame)

# LightGBMのパラメーターを取得します。
params = load_params() | {'learning_rate': 0.01}

# 機械学習します。
cv_result = lgb.cv(params, lgb.Dataset(xs, label=ys), return_cvbooster=True, num_boost_round=2000)
model = cv_result['cvbooster']

# 重要な特徴量を出力します。
print(pd.DataFrame({'feature': model.boosters[0].feature_name(), 'importance': np.mean(model.feature_importance(), axis=0)}).sort_values('importance', ascending=False).head(n=20))

# 学習曲線を出力します。
plot.plot(cv_result['l2-mean'])
plot.show()

# モデルを保存します。
save_model(model)
save_categorical_features(categorical_features)
~~~

今回はモジュール化をしましたので、Titanicの時と比べてコードは単純です。説明する場所がない……。

## 解答作成

保存したモデルを使用して、解答を作成します。

~~~python
import numpy as np
import pandas as pd

from dataset import get_test_data_frame, get_xs
from model import load_categorical_features, load_model


# モデルを読み込みます。
model = load_model()
categorical_features = load_categorical_features()

# データを取得します。
data_frame = get_test_data_frame()
xs = get_xs(data_frame, categorical_features)

# 提出量のCSVを作成します。
submission = pd.DataFrame({'Id': data_frame['Id'], 'SalePrice': np.mean(model.predict(xs), axis=0)})
submission.to_csv('submission.csv', index=False)
~~~

これですべての作業が完了しましたから、Kaggleに提出してみます。スコアは0.12238で329位でした。

![House Pricesのスコアのヒストグラム](./image/score-histgram-house-prices.png)

うん、そこそこ上位ですね（House Pricesのスコアは誤差ですから、小さいほど良い値です）。特徴量の分布とかを丁寧に調べている統計屋さんを、勾配ブースティングのパワーだけで蹴散らしちゃった感じ。プログラミング能力だけでここまで来れるなんて本当に良い時代です。もっと上を目指すならデータ分析が必要なのでしょうけど、このくらい予測できればビジネスでは十分役に立つんじゃないかな。

ほら、ビジネスに機械学習を活用してみたくなってきたでしょ？　実案件でLightGBMするときには、このHouse Prisesのコードを出発点にすればそこそこ楽ちんにプログラミングできると思いますよ。
