Prompot

# Task 1: Research reframing (text only, no LaTeX yet)

Please help me reframe my thesis at a conceptual level.

Context:
- Original V0 tried to imitate Tamn et al. and compare backend embeddings + collaborative filtering.
- New V1 focuses ONLY on backend audio embeddings.
- We compare different model families (e.g., CNN-based, RNN-based, others).
- Target domain is classical music (iPalpiti archive), not general pop music.
- Evaluation is done via proxy tasks and ranking metrics (NDCG, Precision@K, Recall@K).

Deliverables:
1. A refined problem statement
2. 1–2 clear research questions
3. Expected contributions of this thesis (bullet points)

Please keep this aligned with an MVP Master’s thesis.



# Task 2: Literature review restructuring

Task 2: Literature Review Restructuring (V0 → V1)

Context:
I already have a literature review written for V0.
/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Literature_Review

I believe the original version included collaborative filtering, user data,
and imitation of Tamm et al.

In V1, the thesis scope is narrowed to:
- Backend-only audio embedding comparison
- Classical music domain (iPalpiti archive)
- Proxy tasks and ranking-based evaluation (NDCG, Precision@K, Recall@K)

Goal:
I want to keep as much of the existing literature review as possible after year of 2020,
but reinterpret and reorganize it to support the new V1 scope.

Please do the following:
1. Identify which parts of the existing literature review still
   directly support backend audio embedding comparison.
2. Identify which parts should be:
   - de-emphasized,
   - reframed,
   - or moved to background/related work
   (e.g., collaborative filtering, user-based recommendation).
3. Propose a revised literature review structure with section titles,
   ordered logically for the V1 thesis.

Important constraints:
- Do NOT write full paragraphs.
- Do NOT add new citations.
- Focus on structure and rationale only.
- Clearly explain *why* each section belongs in the new structure.







Task 3: Thesis structure for V1 (chapter-level)

Please propose a chapter-level structure for my V1 thesis.

Constraints:
- Follow a standard CS Master’s thesis format.
- Emphasize backend system design + embedding evaluation.
- Methodology should be written as a placeholder, assuming future expansion.
- Results may be partial (small number of songs/models).

For each chapter, include:
- Purpose of the chapter
- What is finalized vs what is provisional (V1)

No LaTeX code yet.




Task 3.5: Chapter 3 skeleton (text only)

Before drafting Abstract or Conclusion, please help me
write a very lightweight skeleton for Chapter 3.

Please include:
- One-paragraph description of the problem setting
  (items, queries, relevance for proxy tasks).
- A high-level description of the backend pipeline
  (audio → embeddings → index → ranking → evaluation).
- Clear statement of what is fixed in V1 and what is provisional.

Constraints:
- No equations.
- No figures.
- No LaTeX.
- Keep it concise and conceptual.





Task 4: Draft key sections for V1

Based on the refined problem statement, research questions,
literature positioning, and the Chapter 3 problem formulation,
please draft the following sections for my V1 thesis:

1. Abstract
2. Introduction
3. Conclusion (V1-level, preliminary)

Assumptions:
- Experiments are small-scale but methodologically sound.
- Results are interpreted cautiously.
- The thesis emphasizes backend design and embedding evaluation,
  not full end-to-end user modeling.
- Scalability and extension are framed as future work.

Tone:
- Honest and academic.
- Clear about limitations.
- Appropriate for a CS Master’s thesis.







Task 5: Methodology placeholder

Please draft a Methodology section as a structured placeholder.

Include:
- Overall pipeline description
- Candidate model families (CNN, RNN, others)
- Embedding generation
- Similarity search / ranking
- Evaluation metrics
- Explicit notes on what is fixed vs what will be expanded later

This section should read as “designed but not fully executed yet”.



## Write to Latex


Task 6-1: Write LaTeX for Chapter 1 (Introduction)

Please write LaTeX source code for Chapter 1 (Introduction)
using the Cal Poly CSC thesis template.

v1 latex file
/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Draft/v1/V1-latex


Constraints:
- Write ONLY the contents of a chapter file (e.g., introduction.tex).
- Do NOT include documentclass, packages, or preamble.
- Start with \chapter{Introduction}.
- Use \section and \subsection where appropriate.
- Do NOT modify or reference main.tex.
- Do NOT invent figures, tables, or citations.
- Preserve the finalized V1 Introduction text exactly in meaning.

Output:
- Raw LaTeX code only.

The abstract and Introduction file.

/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Draft/v1/output/task4.md





Task 6-2: Write LaTeX for Chapter 3 (Problem Formulation and System Overview)

Please write LaTeX source code for Chapter 3
based on the finalized V1 Chapter 3 text.

Constraints:
- Write ONLY the contents of a chapter file (e.g., 03_problem_formulation.tex).
- Start with \chapter{Problem Formulation and System Overview}.
- Use \section and \subsection for Sections 3.1, 3.2, and 3.3.
- Preserve the wording and meaning exactly as provided.
- Do NOT add equations, figures, tables, or citations.
- Do NOT modify or reference main.tex or the preamble.






Task 6-3: Write LaTeX for Chapter 4 (Methodology)

Please write LaTeX source code for Chapter 4
based on the finalized Methodology placeholder.


Constraints:
- Write ONLY the contents of a chapter file (e.g., 04_methodology.tex).
- Start with \chapter{Methodology}.
- Use \section and \subsection as appropriate.
- Preserve the "designed but not fully executed" tone.
- Do NOT invent hyperparameters, results, or implementation details.
- Do NOT add equations, figures, or tables.
- Do NOT modify or reference main.tex or the preamble.









Task 7-1: Write LaTeX skeleton for Chapter 2 (Literature Review)

Please write a LaTeX skeleton for Chapter 2
based on the revised V1 literature review structure.

Constraints:
- Write ONLY the contents of a chapter file (e.g., 02_literature_review.tex).
- Start with \chapter{Background and Literature Review}.
- Include ONLY \section and \subsection titles.
- Do NOT include paragraph text.
- Do NOT include citations or references.
- Do NOT modify or reference main.tex or the preamble.








Task 7-2: Migrate Literature Review Content into Chapter 2

Please migrate content from the following Markdown file
into the corresponding sections of Chapter 2 (Literature Review),
based on the V1 structure.

Source:
literature_review_v2.md
/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Literature_Review/literature_review_v2.md

Constraints:
- Preserve technical meaning and citations.
- Adapt formatting to LaTeX (\section, \subsection, \cite{}).
- Do NOT rewrite or summarize content.
- Minor wording adjustments for flow are allowed.
- Do NOT add new citations.
- Do NOT touch other chapters.

