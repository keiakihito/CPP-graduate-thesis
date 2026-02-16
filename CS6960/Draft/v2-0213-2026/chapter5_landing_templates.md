# Chapter 5 Interpretation Templates

## Capacity--Performance Outcome Scenarios

Common Setup: - Dataset: Small-scale classical archive (iPalpiti) -
Proxies: Composer (Sanity) / Character (Primary) - Models: CNN-Small →
Transformer-Base → Transformer-Large - Metric Example: NDCG@10

------------------------------------------------------------------------

# World A: Proxy-Dependent Behavior (Condition-Dependence Confirmed)

### Hypothetical Results

Composer: 0.70 → 0.84 → 0.90\
Character: 0.61 → 0.62 → 0.61

### Interpretation Template

Large-capacity models improve retrieval performance for structurally
grounded proxies (Composer), but gains saturate for abstract
musical-character retrieval.\
This indicates that the capacity--performance relationship is
proxy-dependent under small-scale, single-domain conditions.

Capacity is therefore not universally beneficial; rather, its utility
depends on task semantics and representational alignment.

------------------------------------------------------------------------

# World B: Monotonic Improvement Across Proxies (Capacity Dominant)

### Hypothetical Results

Composer: 0.70 → 0.84 → 0.90\
Character: 0.58 → 0.71 → 0.82

### Interpretation Template

Under the tested conditions, increased model capacity and large-scale
pretraining strategies consistently improved retrieval performance
across both structural and abstract proxies.

This suggests that, even in small-scale archives, modern large-scale
pretrained models provide transferable representational advantages.

The results indicate that capacity and contemporary pretraining regimes
may act as dominant drivers of retrieval effectiveness within the
defined domain constraints.

------------------------------------------------------------------------

# World C: Performance Plateau (Diminishing Returns)

### Hypothetical Results

Composer: 0.78 → 0.79 → 0.78\
Character: 0.61 → 0.60 → 0.62

### Interpretation Template

Within the tested capacity range, increasing parameter count did not
yield measurable retrieval gains.

This outcome suggests either: - diminishing returns under small-scale
conditions, - limited proxy resolution, - or insufficient dataset
diversity to expose capacity differences.

Rather than demonstrating that capacity is ineffective, the results
delineate the boundary conditions under which capacity effects become
empirically observable.

------------------------------------------------------------------------

# World D: Performance Degradation (Capacity Mismatch)

### Hypothetical Results

Composer: 0.80 → 0.78 → 0.73\
Character: 0.64 → 0.60 → 0.55

### Interpretation Template

Higher-capacity models underperformed relative to compact baselines.

This suggests that increased architectural complexity and large-scale
pretraining do not automatically translate into domain-aligned
similarity representations.

Potential contributing factors include: - representational mismatch
between pretraining objective and proxy definition, - increased
sensitivity to domain-specific noise, - embedding pooling or
segmentation inconsistencies.

This outcome reinforces that capacity alone is not a sufficient
deployment criterion in constrained archival settings.

------------------------------------------------------------------------

# Universal Closing Paragraph (Safe for All Outcomes)

In a small-scale classical music archive, the relationship between
embedding-model capacity and retrieval performance was found to be
\[proxy-dependent / consistently positive / negligible / negative\]
under the tested conditions.

These findings indicate that parameter count alone is not a sufficient
decision criterion.\
Instead, retrieval effectiveness depends on the interaction between
capacity, architectural inductive bias, pretraining objective, proxy
definition, and dataset constraints.

Accordingly, model selection for domain-specific music archives should
prioritize task alignment and representational compatibility rather than
capacity scaling in isolation.
