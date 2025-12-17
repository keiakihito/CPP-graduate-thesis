# Thesis Prototype Compilation Plan

This document outlines the step-by-step process to transform the current Markdown-based proposal and literature review into a compiled PDF thesis using the Cal Poly LaTeX template.

## 1. Content Preparation (Markdown to LaTeX Conversion)

We will convert the existing Markdown content into LaTeX fragments suitable for inclusion in the `chapters/` directory.

-   **Introduction (`chapters/chapter01.tex`)**
    -   Source: `Proposal/proposal_draft.md`
    -   Sections to extract:
        -   Introduction
        -   Research Goal (Problem Statement, Objective, Research Questions)
    -   Action: Convert MD to LaTeX, fix section headers (`\chapter`, `\section`), and ensure citations are formatted as `\cite{key}`.

-   **Literature Review (`chapters/chapter02.tex`)**
    -   Source: `Proposal/Literature_Review/literature_review_v1.md`
    -   Sections to extract: Full content.
    -   Action: Convert MD to LaTeX. Since this file already has citations like `[1]`, we will map them to BibTeX keys (e.g., `[1]` -> `\cite{tamm2024}`).

-   **Methodology & Architecture (`chapters/chapter03.tex`)**
    -   Source: `Proposal/proposal_draft.md`
    -   Sections to extract:
        -   Methodology (Phase 1-3)
        -   System Architecture
        -   Theoretical Foundations
    -   Action: Convert MD to LaTeX. Include the architecture diagram image (ensure `architecture_diagram.png` is moved to `figures/`).

-   **Evaluation Plan (`chapters/chapter04.tex`)**
    -   Source: `Proposal/proposal_draft.md` (Evaluation Roadmap)
    -   Action: Create a short chapter outlining the planned evaluation metrics and roadmap.

## 2. Bibliography Management

We need to create a valid BibTeX file from the reference lists in the Markdown files.

-   **Target File**: `Proposal/Literature_Review/Cal_Poly_Thesis_Template/bibliography/references.bib`
-   **Action**:
    1.  Parse references from `proposal_draft.md` and `literature_review_v1.md`.
    2.  Generate BibTeX entries for each (using MCP/Search to get correct metadata if needed).
    3.  Assign keys (e.g., `tamm2024`, `pons2019`) and ensure they match the `\cite{}` calls in the tex files.

## 3. Template Configuration

We will configure the main LaTeX structure to reflect the project details.

-   **Front Matter (`frontmatter/information.tex`)**:
    -   Update Title: "Comparative Analysis of Pretrained Audio Representations for Small-Scale Music Archives"
    -   Update Author: Keita Katsumi
    -   Update Degree/Department info.
-   **Abstract (`frontmatter/abstract.tex`)**:
    -   Copy Abstract from `proposal_draft.md`.
-   **Chapter Outline (`chapters/outline.tex`)**:
    -   Update `\input{}` commands to include `chapter01`, `chapter02`, `chapter03`, `chapter04`.

## 4. Image Handling

-   **Architecture Diagram**:
    -   Move `Proposal/architecture_diagram.png` (or `.jpg`) to `Proposal/Literature_Review/Cal_Poly_Thesis_Template/figures/`.
    -   Ensure `chapter03.tex` uses `\includegraphics{architecture_diagram}`.

## 5. Compilation

-   **Command**: Run `latexmk -pdf main.tex` inside the template directory.
-   **Validation**: Check for errors (missing citations, image paths) and verify the PDF output.

## 6. Execution Checklist

- [ ] Create `chapter01.tex` (Intro)
- [ ] Create `chapter02.tex` (Lit Review)
- [ ] Create `chapter03.tex` (Methodology)
- [ ] Create `chapter04.tex` (Evaluation)
- [ ] Generate `bibliography/references.bib`
- [ ] Move images to `figures/`
- [ ] Update `frontmatter/information.tex` & `abstract.tex`
- [ ] Update `chapters/outline.tex`
- [ ] Compile PDF

