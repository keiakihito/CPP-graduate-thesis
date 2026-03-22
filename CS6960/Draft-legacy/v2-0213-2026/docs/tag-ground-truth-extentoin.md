Plan: Perceptual Tagging（音響・質感ベース）導入ステップ
Step 0: 目的と制約を固定
- 目的：主観的 “音響・質感” 類似を ground truth（proxy）化し、embedding retrieval を評価する
- 制約：180 tracks、multi-label、graded relevance、K=10（当面）


Step 1: タグ設計（3〜4個に固定）
タグ候補を出す（まずは 6〜8 個）
その中から 4個に絞る（相互に意味が近すぎない・判断可能）
各タグに 1行定義 + 典型例 + 非典型例 を作る（アノテーションの再現性）
例（最終候補イメージ）
- Heavy（重い・密度・低域・強いアタック）
- Bright（明るい・高域が目立つ・軽快）
- Lyrical（歌うよう・滑らか・フレーズ感）
- Dramatic（強いコントラスト・緊張→解放）

Step 2: アノテーション仕様（ルールを決める）
multi-label：複数付与OK
各曲に対して、各タグを 0/1（付与/非付与）で判断（まずはこれが最強に運用しやすい）
もし迷うなら補助的に confidence（低/中/高） を付けても良い（ただし必須にしない）

Step 3: 聴取プロトコル（作業を現実化）
各曲「全曲」ではなく、固定の聴取窓を定義（例：0:30–1:30 + 途中1分）
長尺の “混合” を扱うための最低限の工夫
1曲あたりの最大作業時間を固定（例：2分以内）

Step 4: データ構造（track table に tag column を追加）

最終形（例）：

tag_heavy (0/1)
tag_bright (0/1)
tag_lyrical (0/1)
tag_dramatic (0/1)

（JSONなら tags: ["heavy","lyrical"] でもOKだが、集計と評価が楽なのは列）


Step 5: Graded relevance の定義（評価可能にする）
クエリ曲 q と候補曲 d の relevance を
共有タグ数で graded にする（シンプルで強い）

例：
rel(q,d) = |Tq ∩ Td|（0〜4）
または正規化：rel(q,d) = |Tq ∩ Td| / |Tq|
→ この rel を NDCG の “gain” に使う（0,1,2,3,4 が出せる）


Step 6: “ground truth ranking” を作る
各クエリ q について corpus 内の全曲に rel を計算して、
rel が高い順を “理想ランキング” とみなす
embedding 検索のランキングと比較して NDCG@10 を計算



論文への配置（あなたの案でOK）
Chapter 2（LR）
perceptual / affective attribute のアノテーションが MIR で使われる
subjective label を proxy にして downstream 評価する流れがある

Chapter 3（Objective）
本研究は “perceptual-character proxy” を導入し、multi-label + graded relevance で評価する

Chapter 4（Methodology）
タグ定義（表で1行ずつ）
アノテーション手順（聴取窓、multi-labelルール）
relevance 関数と NDCG の扱い

Chapter 5（Results）
Composer proxy と Perceptual proxy の両方の結果表

“Unlike large-scale datasets …” の一文は Chapter 4のmethod導入 か Chapter 5の結果解釈の導入 に置くのが綺麗です。

英語
Chapter 2: Prior work has shown that perceptual and affective attributes can be used as evaluation proxies in MIR. Chapter 
3: This thesis adopts a perceptual-character proxy using multi-label annotation. 
Chapter 4: Four perceptual tags (Heavy, Light, Lyrical, Dramatic) were defined based on domain-specific considerations...

Unlike large-scale datasets with predefined affective tags, this study constructs a small-scale, domain-specific perceptual annotation to evaluate embedding-based retrieval under constrained archival conditions.

