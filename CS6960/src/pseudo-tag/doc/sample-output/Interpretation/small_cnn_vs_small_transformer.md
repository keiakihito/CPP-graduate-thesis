# 今回の結果はこうまとめられます。

1. small Transformer と small CNN はどちらも similarity space がやや compressed
2. つまり、クラシック曲間の fine-grained distinction はまだ弱い
3. その中でも CNN_small は Transformer_small より ranking quality が高い
4. したがって、この small-capacity 条件では Transformer が優位とは言えない
4. むしろ CNN の局所的特徴抽出の方が、この小規模 classical archive には合っている可能性がある

これはあなたの thesis の仮説にかなり合っています。
「大きい方が常に良い」とはまだ言えず、少なくとも small 同士では CNN が勝っているわけです。



## より大きいモデルなら、

-　embedding space の分離が改善する
-　similarity score の分布が広がる
-　relevant items を上位に押し上げやすくなる

可能性があります。
ただし、それは まだ仮説 です。

## 論文向けの一段落

そのまま使いやすい形にすると、こんな感じです。

The small CNN and small Transformer models both produced highly concentrated similarity scores, suggesting limited separation in the embedding space for fine-grained distinctions among classical tracks. However, while both models achieved the same Precision@5, the CNN model yielded a substantially higher NDCG@5 (0.743 vs. 0.630), indicating that it ranked relevant items closer to the top positions. This suggests that, under the current small-capacity setting, CNN-based embeddings may better preserve retrieval-relevant structure than the small Transformer baseline.

# 一番短い結論

はい、両方とも距離はやや潰れています。
ただ、その中でも CNN_small は Transformer_small より relevant items を上位に置けていて、NDCG で明確に勝っています。
そして次の問いは、まさにあなたが言った通り、base / large にすると score distribution と ranking quality がどう変わるかです。