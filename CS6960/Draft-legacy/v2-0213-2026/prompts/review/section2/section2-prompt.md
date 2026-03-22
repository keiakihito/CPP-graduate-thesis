# prompt 1 for review
I want you to critically review the following Literature Review (Chapter 2).
/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Draft/v2-0213-2026/V2-latex/chapters/chapter02.tex

Context:
My thesis investigates the impact of embedding-model capacity on content-based music similarity retrieval in a small-scale classical music archive (iPalpiti).

Core research structure:
- Hypothesis: Increasing embedding-model capacity does not necessarily improve retrieval performance in small-scale, single-domain settings.
- RQ1: How does model capacity influence retrieval performance?
- RQ2: Under what conditions do higher-capacity models provide meaningful gains?

The thesis follows a hypothesis-driven, research-TDD-inspired design where model capacity is treated as the primary independent variable.

Your task:
1. Check whether Chapter 2 clearly supports this capacity-centered research direction.
2. Identify any sections that drift toward a generic “evaluation theory” or “model survey” narrative rather than supporting the capacity-focused gap.
3. Evaluate whether the Synthesis and Research Gap section logically connects large-scale assumptions to the small-scale capacity question.
4. Identify conceptual misalignments between Chapter 1 and Chapter 2.
5. Suggest concrete improvements only where alignment is weak.

Focus strictly on conceptual alignment and research framing. Do not rewrite stylistically.


# Prompt 2 for grounding
Now perform a grounding and citation integrity check on the same Chapter 2.

Your task:
1. Identify any claims that appear stronger than what the cited papers likely support.
2. Flag statements that generalize beyond the typical scope of MIR or audio embedding literature.
3. Check whether claims about "large-scale performance improvements" or "capacity trends" are appropriately grounded.
4. Suggest where additional citations may be required.
5. Indicate if any statements are potentially overgeneralized or under-cited.

Do not rewrite the text. Focus strictly on grounding and citation robustness.


## process
- PDFs -> text-> clean text with pdftotext in CLI
- List up claims in literature review

### prompot 
Below is my Chapter 2 (raw LaTeX).

Your task:
1. Extract all sentences that contain citations.
2. Convert them into clean claim statements.
3. For each claim:
   - Quote the original sentence.
   - List the cited paper(s).
   - Rewrite the core claim in one simple sentence.
Do not evaluate yet.
Only extract and structure.



