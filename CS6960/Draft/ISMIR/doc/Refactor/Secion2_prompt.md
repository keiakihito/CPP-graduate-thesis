You are a strict ISMIR reviewer focused on clarity, conciseness, and proper scholarly grounding.

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
- Keep all core arguments intact

CORE CLAIMS THAT MUST BE PRESERVED:
- Pretrained audio embeddings are effective but under-evaluated in RecSys contexts
- Capacity scaling is well-studied in large datasets but unclear in small, domain-specific archives
- CNN vs Transformer contrast (local vs long-range representation)
- Evaluation mismatch: classification ≠ retrieval quality
- Metric sensitivity: Recall/F1 vs NDCG/Hit@K under multi-label conditions
- Small-scale / archival constraints affect evaluation validity

STRUCTURE (must keep this order):
1. Problem context (content-based recommendation in small archives)
2. Model architectures (CNN, Transformer, hybrids as background)
3. Evaluation methodology (metrics, limitations, multi-label issues)
4. Recommender systems scope (what is included/excluded)

STYLE:
- ISMIR paper style (concise but precise)
- Each paragraph should have ONE clear role
- Prefer fewer, stronger sentences
- Avoid repeating the same idea in different words

IMPORTANT:
- You are NOT summarizing; you are compressing
- Do NOT remove nuance related to evaluation or task-dependence
- Do NOT weaken technical precision

OUTPUT:
- Return only the rewritten LaTeX section
- Keep LaTeX formatting intact

Here is the text: