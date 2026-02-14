🔥 全体戦略：今やるべきことは3段階
🥇 Phase 1：実験“構造”を完全に固定する（数値は空でOK）

今やるべきことは：

数値を入れるのではなく、
「どの表・どの図・どの比較を出すか」を固定する

つまり：

実験設計は完成

表の枠は完成

図のタイトルも完成

数値だけ空欄

これが理想状態。




🥈 Phase 2：Small sanity experiment を Chapter 4 に組み込む

今ある：

10 songs × MERT

これは超重要です。

なぜなら：

pipeline が動くことの証明

metric が計算できることの証明

proxy が機能することの証明

これは Chapter 4 の

4.1 Pilot Study / Sanity Check

として書けます。


🥉 Phase 3：Chapter 5 は「構造だけ完成」

Chapter 5 は今は：
5.1 Capacity Comparison Results
5.2 Proxy-Specific Results
5.3 Failure Cases
5.4 Overfitting Analysis


のような構造だけ完成させる。

表は：
Table 5.1: NDCG@10 comparison across models
Table 5.2: Composer proxy results
Table 5.3: Character proxy results


# Step 1️⃣ Chapter 4 を完全なTDD設計にする

Chapter 4 に必要なのは：

4.1 Dataset Description

N = [placeholder]

composers = [placeholder]
character labels = [placeholder]
4.2 Models Compared

MERT (size = X params)

CNN-small (X params)

CNN-large (X params)

Transformer-large (X params)

4.3 Experimental Protocol

Leave-one-out retrieval

Similarity = cosine

Metrics = NDCG@K (K = [placeholder])

4.4 Evaluation Procedure

For each query:

exclude self

rank D \ {x_q}

compute metrics

average over queries

4.5 Pilot Validation (10-song experiment)

pipeline sanity check

metric sanity check

no claims

ここまで書いておく。




# Step 2️⃣ Chapter 5 は「分析の型」を決める

ここが重要。

Chapter 5 では：

5.1 Does Capacity Improve Retrieval?

Table: NDCG vs Parameter count

Figure: NDCG vs log(|θ|)

5.2 Intra-family vs Inter-family Comparison

small CNN vs large CNN

small Transformer vs large Transformer

5.3 Proxy Sensitivity

Composer vs Character

差分グラフ

5.4 Overfitting Signals

variance across queries

performance collapse at high capacity?

今は：

数値は全部空でOK。

構造だけ固定。


# Step 3️⃣ Chapter 6 は「条件付き主張テンプレ」を書く

Chapter 6 にはこう書ける：

If higher-capacity models do not significantly outperform smaller models,
this suggests diminishing returns under constrained archival conditions.

Alternatively, if capacity improves retrieval for certain proxy tasks,
this indicates task-dependent utility of model scale.

つまり：

両方の分岐を書いておく。

数値だけ後から入れる。