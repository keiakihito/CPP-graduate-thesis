あなた向けのおすすめ方針

iPalpiti のクラシックで thesis を破綻なく進めるなら、最初の musical character tags はこうするのが良いです。

案1: MER寄りで安全

calm

energetic

tense

sad / dark

joyful / uplifting

これは既存 emotion literature に寄せやすいです。

案2: あなたの研究目的寄り

bright

heavy

lyrical

dynamic

calm

tense

これは retrieval には向いていますが、bright / heavy / lyrical は既存 MER の標準タグ体系から少し外れるので、定義と pilot annotation がより重要です。

私なら、最初の thesis 版は hybrid にします。
つまり:

calm

energetic

tense

lyrical

bright

heavy

この6個です。
前半3つは emotion research に接続しやすく、後半3つはクラシックらしい知覚軸として使えます。

最終的にどう決めるか

おすすめの実装フローはこれです。

20曲だけサンプルを選ぶ

Music2Emo をかけて predicted_moods, valence, arousal を保存する

Librosa で centroid / RMS / tempo / dynamic range を出す

その表を見ながら、あなたが人手で 6タグを貼る

どのタグが安定して貼れるかを見る

安定しないタグは捨てる

この時点で、

calm は貼りやすい

heavy は意外と難しい

lyrical はクラシックでは useful
みたいな感触が出ます。

その結果で最終タグ集合を固定するのが一番きれいです。










あなた向けに一番自然な進め方

おすすめはこの順です。

Phase A

まず 20曲だけ選ぶ
バラけるように

落ち着いた曲

激しい曲

ピアノ独奏

オーケストラ

室内楽
を混ぜる

Phase B

Music2Emo をかけて

predicted_moods

valence

arousal
を出す

Phase C

必要なら librosa で

spectral centroid

RMS

tempo

dynamic range
を出す

Phase D

最終タグを自分で貼る


Q: なぜこのタグ？

A:

1 music emotion literatureで一般的
2 classical repertoireでも意味がある
3 pilot annotationで安定して使えた


私のおすすめはこれです。

calm
tense
energetic
lyrical
bright
heavy


