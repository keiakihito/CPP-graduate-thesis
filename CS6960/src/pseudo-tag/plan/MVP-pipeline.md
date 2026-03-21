おすすめ順

20曲を固定
4タグだけ先に仮決め
人手で 20曲に仮ラベル付け
small CNN / Transformer で embeddings 作成
proxy task 実装
metric 算出
その後に Music2Emo / Librosa を補助証拠として追加


Week 1 の最小ゴール

20曲固定
4タグで人手ラベル
small CNN / Transformer の埋め込み作成
cosine similarity で top-k retrieval
NDCG@5 / Precision@5 / Recall@5 を出す

Week 2

Music2Emo と Librosa を annotation support として追加
6タグ案を見直す
ambiguous tag を削る
error analysis を書く





Tag

energetic (Eerola and Vuoskoski)
tense (Eerola and Vuoskoski)
calm (Eerola and Vuoskoski)
lyrical (ambiguas) <- Define for this task based on


Prior work highlights the ambiguity of categorical emotion labels 
(e.g., calm, relaxed, mellow), which often overlap in meaning 
(Juslin & Laukka, 2004).

To address this while enabling retrieval-based evaluation, we 
introduce a small set of operational tags. In particular, we 
reinterpret the low-to-moderate arousal region—where such 
ambiguities are most prominent—as "lyrical," reflecting 
musically meaningful characteristics.


Since no discrete labels are available, we derive pseudo ground-truth 
labels by partitioning the valence–arousal space based on established 
affective models.

These labels are not directly taken from prior work, but are 
operationalized for the purpose of retrieval evaluation.


論文全体の流れ
1. Content-based retrieval system
2. Embedding models with different capacities
3. Hypothesis: larger models may not generalize well in small domains
4. Two proxy tasks:
   - Composer retrieval (sanity check)
   - Character-based retrieval (main)
5. Pseudo-labels derived from valence–arousal space
6. Ranking metrics (NDCG, Precision@K, etc.)
7. Compare models