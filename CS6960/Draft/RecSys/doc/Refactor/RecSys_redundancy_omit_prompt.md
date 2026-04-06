You are refactoring a RecSys paper for conciseness WITHOUT changing its scientific claims.

Your task is to analyze the DISCUSSION section and identify redundancy, overlap, and unnecessary repetition.

IMPORTANT CONSTRAINT:
You MUST preserve all core scientific claims, system implications, and argument structure.

Use the following invariants as strict tests that must remain true after any refactoring:

# Paper invariants

## Core claim
- The paper must still argue that capacity scaling is task-dependent.
- The paper must still state that larger models do not consistently outperform medium-sized alternatives.
- The paper must preserve that larger models incur higher computational cost.

## Capacity behavior
- The paper must preserve that increasing model capacity does not lead to monotonic performance improvement.
- The paper must allow for cases where medium-sized models outperform larger models.

## Task framing
- Composer retrieval must remain the structured proxy.
- Musical-character retrieval must remain the abstract semantic proxy.

## Metric interpretation
- Recall/F1 at small K with large relevant sets must be described as less informative.
- NDCG/Hit@K must remain the more stable ranking-based indicators.

## Experimental validity
- The paper must preserve that model capacity is explicitly scaled (e.g., parameter size).
- The paper must preserve that the embedding pipeline (e.g., segmented processing) is realistic.

## System implications (RecSys-specific)
- The paper must still argue that larger models do not provide consistent performance gains relative to their cost.
- The paper must preserve the notion of diminishing returns from scaling.
- The paper must preserve that efficient / mid-sized models can be preferable in practice.

## Section roles
- The Discussion section must interpret results without introducing new experimental findings.
- The Discussion must clearly connect results to system design implications.

## Scope constraints
- The paper must remain a small-scale classical archive case study.
- The paper must not overclaim generalization beyond this setting.

---

INSTRUCTIONS:

1. Identify redundant sentences, repeated claims, or overlapping explanations.
2. Pay special attention to:
   - repeated mentions of "capacity scaling is task-dependent"
   - repeated explanations of metric limitations (Recall/F1 vs NDCG)
   - repeated system implications (efficiency vs large models)
3. Do NOT remove anything that carries unique meaning.
4. Prefer compression and merging over deletion.
5. Be conservative: if unsure, do NOT suggest removal.

---

OUTPUT FORMAT (strict):

For each issue:

[Redundant Passage]
(paste the exact sentence or sentences)

[Why redundant]
(explain briefly, e.g., repetition, overlap, rephrasing of same claim)

[Safe Refactor Suggestion]
(provide a shorter or merged version in polished academic English)

[Invariant Check]
(state explicitly why all invariants remain satisfied)