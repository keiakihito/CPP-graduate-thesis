# Vibe Research × Software Engineering 忘備録

## 0. 前提スタンス

- 自分の立ち位置は **scientist ではなく software engineer**
- ただし「研究っぽいこと」をやるのではなく  
  **エンジニアリングで research を前に進める**ことを目的とする
- Recommendation System のような
  - 実装
  - 評価
  - 再現性
  が重要な分野は、このスタンスと相性が良い

---

## 1. Unix Pipe 的発想で論文を見る

### Unix Pipe の本質
- 小さいプログラムを作る
- 1つの責務だけを持たせる
- pipe で組み合わせる
- **全体ではなく「pipe の性質」を理解する**

### Research に対応させると

1論文 = 1 pipe


```
Problem
|→ Hypothesis (意図)
|→ Minimal Method
|→ Evaluation
|→ Insight
```


- 1つの問い
- 1つの仮説
- 1つの設定
- 1つの Insight

👉 大きな論文は不要  
👉 **小さくて狭い論文をたくさん作る**

---

## 2. Vibe Research と Software Development の対応

### Research と開発は同型

| Software Development | Research |
|---|---|
| 要件定義 | Research Question |
| 設計意図 | 仮説 / Motivation |
| 実装 | Methodology |
| テスト | Evaluation |
| バグ・想定外 | Experimental Result |
| リファクタ | Discussion |
| 設計理解の更新 | Insight |

---

## 3. TDD と Research の対応

### TDD の流れ
- Stub
- Red
- Green
- Refactor

### Research に置き換えると

- 仮説を立てる（Stub）
- 実験してズレる（Red）
- 条件を調整して成立させる（Green）
- 理解を整理する（Refactor）

👉 **Insight は Refactor に相当する**

---

## 4. 「意図」と「Insight」の関係

### 意図 (Intent)
- 実験前に持っているもの
- 「こうなるはず」
- 「これが効くはず」
- 設計者の仮説

例：
> embedding を正規化すれば cosine similarity なので性能が上がるはず

※ これはまだ Insight ではない

---

### Insight
- 実験後にしか得られない
- **意図が現実にぶつかった後の理解**
- 条件付き・制約付きでの振る舞いの説明

例：
> 正規化は cosine similarity では有効だが、dot product では不安定になる

👉 **意図が更新された状態**

---

### 一言で言うと
> **Insight = 設計意図が現実と衝突したログ**

---

## 5. Insight が「ない」論文とは

- 何をやったかは書いてある
- 数値は出ている
- でも「なぜそうなったか」がない

典型例：
- 結果の列挙だけ
- 先行研究と一致した、で終わる
- 「今後の課題とする」で逃げる

👉 **How はあるが Why がない**

---

## 6. Insight が「ある」状態

- 条件が明示されている
- なぜそう振る舞ったかの仮説がある
- 他の設定に転用できる

最低限これに答えられていれば OK：
- なぜこの条件で効いた？
- なぜこの条件で壊れた？
- どれは「やらなくてよかった」？

---

例（Recommendation / Embedding 系）
❌ Insight なし
Normalized embeddings performed better than unnormalized ones.

✅ Insight あり
Normalization consistently improves retrieval metrics when cosine similarity is used, suggesting that magnitude variance introduces noise rather than signal in this setting.

後者は：

- 条件がある
- 原因仮説がある
- 他の設定に転用できる

## 7. 小さい論文大量生産が成立する理由

- Insight は **大きくなくていい**
- 「1 setting × 1 metric × 1 insight」で十分
- 再利用可能な理解が1つ増えれば価値がある

---

## 8. Cursor / GPT / 自分の役割分担

### 自分がやること
- 問いを決める
- 評価軸を決める
- CLI を叩く
- 結果を見て違和感を持つ

👉 人間の価値はここ

### Cursor に任せる
- Boilerplate 論文生成
- Related Work 下書き
- Methodology 記述
- 図表作成
- 実験ログ整理

### GPT に相談する
- 仮説のシャープ化
- Insight の言語化
- 「これは論文になるか？」の判断
- 次の pipe への接続整理

---

## 9. なぜこれが開発現場に効くのか

- 設計意図を言語化できるようになる
- 「なぜこの選択をしたか」を説明できる
- 実験ログから判断基準を作れる
- 勘ではなく **条件付きの理解** を持てる

👉 結果として：
- 強い設計レビュー
- 強い技術選定
- 強い説明責任

---

## 10. 最終的な指針

- 1論文 = 1 Insight
- Insight は「意図が裏切られたところ」にある
- Research は Software Engineering の延長で考えてよい
- 作ったもの・書いたものが、自分の未来を作る




小さくて狭い論文でも Insight を出すコツ

実践的なチェックリストを置いておきます。論文1本につき、最低1つこれに答えられていれば OK：

- なぜこの設定だと性能が落ちた？
- なぜこの metric では差が出た？
- なぜこのモデルは安定している？
- どの仮定が壊れるとダメになる？
- どの要素は やらなくてよかった？

👉 「やらなくてよかった」も立派な Insight


Vibe Research を Insight 生成器にするコツ あなたのスタイルを壊さずにやる方法です。

実験前にこれを1行書く
I expect that X will happen because Y.

実験後にこれを書く
However, under condition Z, the behavior changed in that ...
