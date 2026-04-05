You are refactoring a research paper for conciseness WITHOUT changing its scientific claims.

Your task is to analyze the DISCUSSION section and identify redundancy, overlap, and unnecessary repetition.

IMPORTANT CONSTRAINT:
You MUST preserve all core scientific claims and argument structure.

Use the following invariants as strict tests that must remain true after any refactoring:

# Paper invariants

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
- The paper must preserve that model capacity is explicitly scaled.

## Section roles
- The Discussion section must interpret results without introducing new experimental findings.

## Scope constraints
- The paper must remain a small-scale classical archive case study.
- The paper must not overclaim generalization.

---

INSTRUCTIONS:

1. Identify redundant sentences, repeated claims, or overlapping explanations.
2. Do NOT remove anything that carries unique meaning.
3. Prefer compression over deletion when possible.
4. Be conservative: if unsure, do NOT suggest removal.

---

OUTPUT FORMAT (strict):

For each issue:

[Redundant Passage]
(paste the exact sentence or sentences)

[Why redundant]
(explain briefly)

[Safe Refactor Suggestion]
(provide a shorter or merged version)

[Invariant Check]
(state why all invariants remain satisfied)