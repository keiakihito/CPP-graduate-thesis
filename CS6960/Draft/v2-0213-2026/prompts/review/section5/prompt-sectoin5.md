# Generate
I am writing Chapter 5 of a master's thesis titled:

"Experimental Results and Analysis"

This chapter must strictly follow the capacity–performance hypothesis defined in Chapter 4.

The goal of this chapter is NOT to compare which model is best, but to analyze how embedding-model capacity behaves across proxy retrieval tasks.

Please generate a structured Chapter 5 draft using the following exact section outline and constraints:

---

5. Experimental Results and Analysis

5.1 Experimental Setup Recap  
5.2 Sanity Proxy Results  
5.3 Primary Musical Proxy Results  
5.4 Capacity–Performance Trends Within Architectural Families  
5.5 Cross-Family Descriptive Comparison  
5.6 Robustness and Confidence Interval Analysis  
5.7 Summary of Empirical Findings  

---

CRITICAL REQUIREMENTS:

1. The structure must emphasize capacity scaling behavior.
2. The results must be presented per model with parameter counts shown.
3. All tables must include:
   - Model name
   - Parameter count
   - NDCG@10 (placeholder)
   - 95% CI (placeholder)
4. Do NOT claim causal superiority between architectural families.
5. Within-family scaling must be explicitly analyzed (small → base → large).
6. Cross-family comparisons must be descriptive only.
7. Bootstrap confidence intervals must be referenced in the robustness section.
8. Interpretation should remain empirical — theoretical explanation belongs to Chapter 6.

For each section, follow these specific design intentions:

---

5.1 Experimental Setup Recap  
- Brief restatement of:
  - Number of tracks
  - Number of models (6 total)
  - Architectural families (CNN / Transformer)
  - Evaluation metrics (NDCG@10, Precision@10)
  - Bootstrap (1,000 iterations)

Keep concise (no repetition from Chapter 4).

---

5.2 Sanity Proxy Results  
- Present composer-based retrieval results.
- Include a full table with placeholder values.
- Add a short paragraph describing:
  - Whether performance increases monotonically within CNN
  - Whether performance increases monotonically within Transformer
- Avoid deep interpretation.

---

5.3 Primary Musical Proxy Results  
- Same structure as 5.2.
- Explicitly note if trends differ from sanity proxy.
- Emphasize whether capacity scaling behaves differently here.

---

5.4 Capacity–Performance Trends Within Architectural Families  
This is the core section.

- Analyze CNN scaling behavior.
- Analyze Transformer scaling behavior.
- Categorize trends as:
  - Monotonic
  - Saturation
  - Degradation
- Explicitly connect observations to the condition-dependent hypothesis.

---

5.5 Cross-Family Descriptive Comparison  
- Compare approximate performance levels across families.
- Use cautious language.
- Emphasize that parameter count, architecture, and pretraining co-vary.
- Do NOT infer architectural superiority.

---

5.6 Robustness and Confidence Interval Analysis  
- Explain how bootstrap CI was used.
- Discuss whether CI overlap changes interpretation.
- Clarify if observed differences are statistically robust.

---

5.7 Summary of Empirical Findings  
- Summarize key empirical observations.
- Do NOT theorize.
- Explicitly state whether results:
  - Support
  - Weaken
  - Complicate
  the capacity–performance hypothesis.

---

Tone:
Academic, precise, non-speculative.
Avoid hype language.
No forward references to Chapter 6 explanations.

Produce LaTeX-ready text including table environments.
Use placeholders like "--" for numerical values.
