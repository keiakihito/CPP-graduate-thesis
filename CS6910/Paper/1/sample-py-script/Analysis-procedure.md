A good prompt for Claude Code to generate an “architecture doc”

Copy-paste this (change anything you like):

Task: Produce a concise Architecture & Replication Guide (markdown) for the repository Darel13712/pretrained-audio-representations.
Audience: Graduate thesis meeting prep (music recommendation).
Scope:

Pipeline architecture: Describe Stage 1 (pretrained embedding extraction) and Stage 2 (recommenders), mapping each pipeline box to specific files/functions in the repo (e.g., extract_item_embeddings/*, preprocess/*, train.py, train_bert.py, knn.py, model.py, bert4rec.py, metrics.py, table.py).

Dataset policy: Summarize the Music4All-Onion split (last month as val/test, previous year train; remove cold users/items), sequence length cap (300), and expected input formats.

Model architectures: In 2–3 sentences each, explain KNN, Shallow Net (frozen item embeddings, learnable user embeddings, cosine + hinge loss, negative sampling), and BERT4Rec (masked sequential transformer with frozen projection).

Training & evaluation: Show the training entry points (train.py, train_bert.py), hyperparameters at a glance (optimizer, losses, negatives), metrics (HitRate@50, Recall@50, NDCG@50), and how metrics.py/table.py aggregate results.

Sanity-check expectations: Note that in the paper BERT4Rec ≥ Shallow Net ≥ KNN, and MusiCNN tends to be strong.

Small-dataset plan: Add a short section on how to substitute a small classical CD dataset + Spotify-enriched metadata and generate mock user logs consistent with the repo’s preprocess step.
Format: A single markdown file with headings, a “Slide-to-Code Map” table, and checklists I can hand to my advisor.

That prompt ensures Claude focuses on architecture mapping rather than general prose.