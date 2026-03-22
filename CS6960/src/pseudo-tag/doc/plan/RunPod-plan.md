# RunPod Setup Plan for Thesis Experiments

This document describes how to run embedding + retrieval evaluation on a
GPU environment using RunPod.

------------------------------------------------------------------------

# 🎯 Goal

Run the following pipeline on GPU:

1.  Build embeddings (CNN / Transformer)
2.  Run batch retrieval evaluation
3.  Compare model performance (NDCG, Precision)

------------------------------------------------------------------------

# 🧠 Strategy

-   Use RunPod for GPU compute
-   Keep local machine for:
    -   development
    -   pseudo-label generation
-   Run heavy embedding extraction on GPU
-   Download results back to local

------------------------------------------------------------------------

# 🚀 Step 1: Create RunPod Instance

1.  Go to: https://www.runpod.io/
2.  Create a Pod

## Recommended GPU (start small)

-   RTX 4090 (best cost/performance)
-   A100 (if needed)

## Settings

-   Template: PyTorch (or base Ubuntu)
-   Storage: 20--50GB
-   Enable SSH

------------------------------------------------------------------------

# 🔑 Step 2: Connect via SSH

ssh root@`<RUNPOD_IP>`{=html} -p `<PORT>`{=html}

------------------------------------------------------------------------

# 📦 Step 3: Clone Repository

git clone `<YOUR_REPO_URL>`{=html} cd `<repo>`{=html}

------------------------------------------------------------------------

# ⚠️ Repo Structure Note

Your current repo:

CS6960/ ├── pseudo-tag/ ├── other stuff...

👉 We will work inside:

cd pseudo-tag

------------------------------------------------------------------------

# 🐍 Step 4: Setup Python Environment

python3 -m venv .venv source .venv/bin/activate

pip install -U pip pip install -r requirements.txt

------------------------------------------------------------------------

# 📁 Step 5: Prepare Data

## Option A (recommended for now)

scp -r data/wav root@`<IP>`{=html}:/workspace/pseudo-tag/data/ scp
data/output/labels/pseudo_labels.csv
root@`<IP>`{=html}:/workspace/pseudo-tag/data/output/labels/

------------------------------------------------------------------------

# 🧪 Step 6: Smoke Test

pytest tests/test_embedding_extractors_smoke.py -m smoke -v

------------------------------------------------------------------------

# ⚙️ Step 7: Build Embeddings

python -m src.pipeline.build_embeddings data/wav data/output/embeddings
--model-name cnn_small

python -m src.pipeline.build_embeddings data/wav data/output/embeddings
--model-name transformer_small

------------------------------------------------------------------------

# 📊 Step 8: Run Batch Evaluation

python -m src.evaluation.run_batch_eval data/wav
data/output/embeddings/transformer_small_embeddings.npy
data/output/embeddings/transformer_small_metadata.json
data/output/labels/pseudo_labels.csv --model-name transformer_small
--top-k 5 --relevance-strategy tag_overlap

------------------------------------------------------------------------

# 💾 Step 9: Download Results

scp root@`<IP>`{=html}:/workspace/pseudo-tag/data/output/embeddings/\*
./local_output/

------------------------------------------------------------------------

# ⚠️ Important Tips

## Use tmux

tmux

## Stop Pod when done

RunPod charges while running

------------------------------------------------------------------------

# 🎯 Summary

Local: - develop code - generate pseudo labels

RunPod: - build embeddings - run evaluation

Result: - download metrics
