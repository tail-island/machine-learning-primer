# 機械学習の手法選択

本稿では、これまで線形回帰、決定木、ランダム・フォレスト、K-meansといういくつかの手法を使用してきました。実は機械学習にはこれ以外にもたくさんの手法があって、それを使い分けなければならない……のは大変すぎるので、私が手法選択に使用しているルールを疑似コードで示しましょう。

~~~python
if 大量のデータを用意できる and (そこそこ高性能なGPUを使える or 潤沢なクラウド予算がある) and テーブル・データではない:
    return 深層学習
else
    return 勾配ブースティング
~~~

……ごめんなさい。実は私は普段は線形回帰とか全くやってません。なんでかというと、私は統計の人じゃなくて機械学習の人だから。

## 統計と機械学習

統計と機械学習って同じ道具を使うことも多くて似ているのですけど、目的が異なります。統計は「説明」を目的とし、機械学習は「予測」を目的とします。気温とアイスクリームの売り上げには関係があるのか、関係があるのであればどのような関係なのかを考えるのが統計で、どんな関係なのかはさておいて予測の精度を高めることにひたすら注力するのが機械学習なんです。

統計をしたいのであれば、データの関係を明らかにしたいですから、モデルはできるだけ単純な方が良い。シンプルなモデルの方が関係が分かりやすいですもんね。だから、データに合致する範囲で、できるだけシンプルなモデルを構築します。だから、まずは線形回帰で関係があるかを調べようとなります。

機械学習では予測精度をできるだけ高めたいので、複雑なモデルであっても、モデルの中身がブラックボックスでデータ間の関係が理解不能であってもお構いなし。だから、できるだけ高い予測精度を期待できるモデルを選びます。よって、線形回帰みたいな精度が低そうなモデルは選択肢にあがらないわけ。

で、私は機械学習の人なので、前に書いたように、現在の技術で最高の精度を誇る深層学習（deep learning）と勾配ブースティング（gradient boosting）しか使わないんですな。

## 深層学習の弱点

いや精度第一ならば深層学習一択だろうという反論が聞こえてきそうですけど、深層学習って弱点も多いと思うんですよ。

その深層学習は脳の構造を模した手法とかよく言われますけど、今どきの深層学習モデルは脳のどこかを模しているわけではありません。独自に進化した結果、脳とはあまり関係なくなっています。

ぶっちゃけて言えば、私は深層学習とは行列演算を大量に積み重ねたもので、その行列の各要素をパラメーターとする、異常に大量のパラメーターを使える機械学習の手法と認識しています。で、パラメーターが大量にあったら調整が大変に思えるのですけど、それは逆誤差伝播法（backprobagation）という適用可能な範囲は狭いけど効率が良い調整方式で対応しちゃう。あと、大量のパラメーターを使って計算するのは時間がかかりそうに思えるけど、それは行列の特性をイイ感じに活用した並列処理で対応しちゃう。

ただね、いくら逆誤差伝播法が効率が良くても、深層学習の良さを出そうとしてパラメーターを大量に使用したなら、やっぱりパラメーターの調整は大変になります。この大変さは、「大量のデータが必要」という形になって我々を苦しめます。

あと、行列の特性をイイ感じに活用した並列処理で高速化といっても、たとえば8コアのCPUだとハイパー・スレッディングを使用しても16並列でしか処理できません。だから、膨大な数の並列処理が可能なGPU（Graphicd Processing Unit）やTPU（Tensor Processing Unit）が必要となります。そこそこ高性能なGPUを買うか、GPUやTPUを使用可能なクラウドを借りるかしなければならないわけで、とにかくお金がかかります。

あと、深層学習ってのは、テーブル・データに弱いです。実は本稿の深層学習のところで述べるTransformerベースならそこそこいけるかもしれないのですけど、少なくとも、他の機械学習の手法と比べて面倒が多い。画像処理や自然言語処理のような、深層学習以外では精度が出ないような場合以外、いわゆるテーブル・データ（表形式のデータ）の場合は深層学習以外の手法を試した方が良いんじゃないかな。

まとめると、大量のデータが必要で、GPUやTPUが必要（ローカルでもクラウドでもよい）で、テーブル・データではいろいろ面倒な癖に今のところ明確なアドバンテージがないんです。

## 機械学習の手法選択

というわけで、大量のデータを用意できないとか、GPUを持っていないしクラウド予算もないとか、テーブル・データであるとかの場合は、私は勾配ブースティングで機械学習することにしているというわけ。