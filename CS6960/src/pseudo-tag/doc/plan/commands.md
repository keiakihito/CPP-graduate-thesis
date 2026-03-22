# Thesis MVP CLI Commands

  This document summarizes the core CLI commands for the thesis MVP pipeline.

  ---

  # 🧠 Pipeline Overview

  The main pipeline consists of three stages:

  1. Build pseudo labels
  2. Build embeddings
  3. Run retrieval evaluation

  ```text
  wav -> pseudo labels -> embeddings -> retrieval -> evaluation
  ```

  ———

  # 1. Build Pseudo Labels

  Generate pseudo labels from valence/arousal predictions.

  ```
  python scripts/build_pseudo_labels.py
  ```
  Output:

  - data/output/pseudo_labels.csv

  ———

  # 2. Build Embeddings

  Generate corpus embeddings for all WAV files.

  CNN embeddings:
  ```
  python -m src.pipeline.build_embeddings data/wav outputs/cnn --model-name cnn_small
  ```

  Transformer embeddings:
  ``` 
  python -m src.pipeline.build_embeddings data/wav outputs/ast --model-name transformer_small
  ```
  Outputs:

  - *_embeddings.npy
  - *_metadata.json
  - *_failures.json

  ———

  # 3. Run Retrieval Evaluation

  Run top-k retrieval for one query and evaluate with pseudo labels.

  CNN example:
  ```
  python -m src.evaluation.run_retrieval_eval \
    data/wav/track_30_Passacaglia.wav \
    outputs/cnn/cnn_small_embeddings.npy \
    outputs/cnn/cnn_small_metadata.json \
    data/output/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy composer
  ```
  
  Transformer example:
  ```
  python -m src.evaluation.run_retrieval_eval \
    data/wav/track_30_Passacaglia.wav \
    outputs/ast/transformer_small_embeddings.npy \
    outputs/ast/transformer_small_metadata.json \
    data/output/pseudo_labels.csv \
    --model-name transformer_small \
    --top-k 5 \
    --relevance-strategy tag_overlap
  ```

  Printed output:

  - top-k retrieved items
  - relevance list
  - precision@k
  - recall@k
  - ndcg@k

  ———

  # 4. Batch Evaluation
  Transformer
  ```
    python -m src.evaluation.run_batch_eval \
    data/wav \
    data/output/embeddings/transformer_small_embeddings.npy \
    data/output/embeddings/transformer_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name transformer_small \
    --top-k 5 \
    --relevance-strategy tag_overlap
  ```

  CNN
  ```
    python -m src.evaluation.run_batch_eval \
    data/wav \
    data/output/embeddings/cnn_small_embeddings.npy \
    data/output/embeddings/cnn_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy tag_overlap
  ```

  ———

  # Notes

  - pseudo_labels.csv should contain track_id, wav_path, composer, and tag columns.
  - Retrieval/evaluation now prefers track_id matching over absolute path matching.
  - Both embedders are inference-only wrappers around pretrained models.

