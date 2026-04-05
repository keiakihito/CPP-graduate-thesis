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

