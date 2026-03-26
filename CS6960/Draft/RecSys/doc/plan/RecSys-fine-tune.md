RecSys-fine-tune.md

What stays exactly the same: Your dataset, chosen models (CNNs and Transformers), proxy tasks (Sanity and Musical Character), and evaluation metrics (NDCG, Precision, Recall) do not need to change. Your core experiments and results remain perfectly valid for RecSys.

What must be added/modified in the Literature Review: You need to add literature that explicitly connects your work to Recommender Systems community challenges. You must add text discussing the "cold-start" and data sparsity problems, positioning your content-based approach as a necessary solution for curated archives that lack dense user interaction logs. You should also add text explicitly pointing out the gap that pretrained audio representations are rarely explored systematically within the context of music recommender systems, using the Tamm & Aljanaki paper to support this claim.


What must be added/modified in the Methodology: While the proxy tasks themselves do not change, you must add text to justify them. Because RecSys audiences expect evaluations based on real user interaction logs, you need to add a defense in your methodology explaining why you are using metadata proxy tasks (composer and musical character) instead. You should explicitly frame these proxy tasks as necessary deterministic, offline baselines that evaluate the intrinsic quality of the backend embeddings before the noise of complex user preference modeling is introduced.


Details

 simple copy-and-paste into the ACM template is not enough. While the underlying methodology and results of your study are perfectly valid for the ACM Conference on Recommender Systems (RecSys), you must actively rewrite and revise specific parts of your manuscript to appeal to the RecSys audience, as well as fulfill the strict formatting requirements of the ACM template.
Here is a breakdown of what you need to revise to fine-tune your paper for RecSys:
1. ACM Template & Formatting Requirements
The ACM template requires specific elements that were not present in your ISMIR draft. Based on the ACM guidelines, you must add the following:

    CCS Concepts (Computing Classification System): You must generate and include XML code for the 2012 ACM Computing Classification System. For your paper, the primary concept should be: Information systems → Recommender systems.
    Keywords: You must provide a comma-separated list of user-defined keywords (e.g., Recommender Systems, Music Recommendation, Pretrained Audio Representations, Cold-Start).
    Document Class: Ensure your main .tex file uses the correct document class and style for conference proceedings. For RecSys, this is typically \documentclass[sigconf]{acmart}.

2. Content Revisions: Framing for the RecSys Community
The RecSys community is fundamentally user-centric, meaning they expect papers to deal with user interaction logs, sequential behaviors, or collaborative filtering matrices. Because your paper evaluates the backend using proxy tasks instead of actual user logs, you must reframe your narrative to justify this approach.
Revise the Introduction (The "Cold-Start" Framing):

    Do not just frame the problem around "small-scale classical music archives." Frame it explicitly around the data sparsity and cold-start problems inherent in highly curated systems.
    Argue that because traditional collaborative filtering fails without dense user interaction data, Content-Based Filtering (CBF) utilizing audio embeddings is the only viable recommendation strategy in these environments.

Revise the Related Work:

    You must explicitly position your work as answering a gap in the recommender systems literature. Use the Tamm & Aljanaki paper to support this: state that while the Music Information Retrieval (MIR) community frequently proposes pretrained backend models, their efficiency and applicability for Music Recommender Systems (MRS) are underexplored.
    Mention that the RecSys community often favors traditional end-to-end neural network learning, making an offline comparative analysis of pretrained representations highly necessary.

Defend the Methodology (Proxy Tasks):

    In Section 3 (Methodology), you must add a brief justification for using proxy tasks (Composer and Musical Character) instead of historical user logs.
    Explain that evaluating the intrinsic representational quality of the embeddings via deterministic, offline metadata proxies is a necessary baseline evaluation before introducing the noise of complex user preference models.

Revise the Implications / Conclusion:

    Ensure your conclusion explicitly speaks to recommender system designers and practitioners.
    Emphasize the practical takeaway: your findings demonstrate that practitioners do not need to blindly deploy computationally expensive, massive capacity models in sparse recommendation environments, as lightweight models provide comparable retrieval performance




Note
Literature review:
今の節で必要なのは主にこの4種類です。

1. 音楽推薦では user feedback の sparsity / cold-start が本質的課題である
2. そのため content-based / content-driven な方法が重要になる
3. 音楽推薦で audio signal は正当な content source である
4. ただし pretrained audio representations の比較は MRS/RecSys ではまだ十分整理されていない