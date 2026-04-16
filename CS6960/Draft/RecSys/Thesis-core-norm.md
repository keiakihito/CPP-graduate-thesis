# Thesis-core-norm / Paper invariants

## Core claim
- The paper must still argue that capacity scaling is task-dependent.
- The paper must still state that larger models do not consistently outperform medium-sized alternatives.

## Capacity behavior
- The paper must preserve the observation that increasing model capacity does not lead to monotonic performance improvement.
- The paper must allow for cases where medium-sized models outperform larger models.

## Task framing
- Composer retrieval must remain the structured proxy.
- Musical-character retrieval must remain the abstract semantic proxy.

## Metric interpretation
- Recall/F1 at small K with large relevant sets must be described as less informative.
- NDCG/Hit@K must remain the more stable ranking-based indicators.

## Experimental validity
- The paper must preserve that model capacity is explicitly scaled (e.g., via parameter size or architectural depth).

## Section roles
- The Results section must report empirical observations without introducing causal explanations.
- The Discussion section must interpret results without introducing new experimental findings.

## Scope constraints
- The paper must remain a small-scale classical archive case study.
- The paper must not overclaim generalization beyond the studied setting.



RecSys論文　core

We show that in cold-start candidate generation, model capacity scaling does not reliably improve ranking quality, 
while significantly increasing ingestion cost. 
Therefore, model selection should be treated as a system-level cost–quality trade-off rather than a pure performance optimization problem.


👉 proxy taskは：
👉 「本来のユーザ評価がないので、代わりにランキングの良さを測る仕組み」

“Proxy tasks are used to approximate relevance in the absence of user interaction data, enabling evaluation of candidate ranking quality in a controlled cold-start setting.”

🚀 まとめ（完全理解版）

論文のコア
👉 大きいモデルはcandidate generationではコストに見合わない場合がある

proxy taskの役割
👉 ranking qualityを測るための代替評価設計

2つをつなぐと
👉
「適切なproxy taskと評価設計のもとで見ると、capacity scalingは必ずしも有効ではない」

💡 最後に（めちゃ重要）
あなたの理解は今👇
👉 ほぼ完成してる

あとはこの違い👇だけ
* ❌ taskが主役
* ⭕ system decision（コスト vs 性能）が主役


🔥 完全版（あなたの言葉で）
👉 これで完璧です👇
👉
「候補生成の段階では、モデルの容量を大きくしてもランキング品質は一貫して改善するわけではなく、タスクによって効果が変わる。
一方でコストは確実に増大するため、モデル選択は性能最大化ではなくコストとのトレードオフとしてシステム設計の中で考える必要がある。」



