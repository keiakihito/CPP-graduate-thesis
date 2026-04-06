You are a strict RecSys / CIKM reviewer focused on system design, evaluation validity, and decision-making relevance.

Your task is to COMPRESS the Related Work section while preserving ALL references and scientific claims.

CRITICAL REQUIREMENTS:
- DO NOT remove or omit ANY citation (all \cite{} must remain)
- DO NOT drop any referenced paper, even if sentences are merged
- DO NOT introduce new citations
- Every existing citation must still appear in the rewritten version

GOALS:
- Reduce length by ~30–40%
- Remove redundancy and repeated claims
- Merge sentences that express similar ideas
- Improve flow and readability
- Emphasize system-level relevance and decision-making implications

CORE CLAIMS THAT MUST BE PRESERVED:
- Content-based retrieval is essential for cold-start and data-sparse recommendation settings
- Embedding models function as candidate generation components in recommender pipelines
- Model capacity introduces computational cost (latency / resource usage) that must be justified
- Capacity scaling is well-studied in large-scale benchmarks but under-explored in small, curated archives
- CNN vs Transformer differences should be framed as architectural trade-offs, not superiority claims
- Evaluation mismatch: classification metrics do not necessarily reflect retrieval or ranking quality
- Ranking metrics (NDCG, Hit@K) are more appropriate under multi-label or dense relevance settings
- Set-based metrics (Recall, F1) can be misleading when relevant items are numerous relative to K
- System constraints (small catalog, no interaction data) fundamentally affect evaluation design and conclusions

STRUCTURE (must keep this order):
1. Problem setting (cold-start, data-sparse recommendation, candidate generation role)
2. Embedding models as system components (CNN, Transformer framed as design choices)
3. Capacity vs cost trade-off (scaling vs efficiency, system implications)
4. Evaluation methodology (ranking metrics, multi-label relevance, metric limitations)
5. Scope within recommender systems (what is evaluated vs not, e.g., no user modeling)

STYLE:
- RecSys / IR paper style (concise, decision-oriented, system-aware)
- Each paragraph should have ONE clear role
- Prefer fewer, stronger sentences
- Avoid repeating the same idea in different words
- Avoid MIR-specific framing unless necessary

IMPORTANT:
- You are NOT summarizing; you are compressing
- Do NOT remove nuance related to evaluation or task-dependence
- Do NOT weaken technical precision

OUTPUT:
- Return only the rewritten LaTeX section
- Keep LaTeX formatting intact

Here is the text: