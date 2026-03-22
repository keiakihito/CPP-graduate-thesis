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
  python -m src.pipeline.build_embeddings data/wav data/output/embeddings/cnn_small --model-name cnn_small
  python -m src.pipeline.build_embeddings data/wav data/output/embeddings/cnn_medium --model-name cnn_medium
  python -m src.pipeline.build_embeddings data/wav data/output/embeddings/cnn_large --model-name cnn_large
  ```

  Transformer embeddings:
  ``` 
  python -m src.pipeline.build_embeddings data/wav data/output/embeddings/transformer_medium --model-name transformer_medium
  python -m src.pipeline.build_embeddings data/wav data/output/embeddings/transformer_large --model-name transformer_large
  ```
  Outputs:

  - *_embeddings.npy
  - *_metadata.json
  - *_failures.json

  ———
  # 2.5 Build embeddings test
  pytest tests/test_build_embeddings.py -v -s
  pytest tests/test_embedding_extractors_smoke.py -m smoke -v -s

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
  Transformer medium
  ```
  python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/transformer_medium/transformer_medium_embeddings.npy \
  data/output/embeddings/transformer_medium/transformer_medium_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name transformer_medium \
  --top-k 5 \
  --relevance-strategy tag_overlap

  ```
   Transformer large
   ```
   python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/transformer_large/transformer_large_embeddings.npy \
  data/output/embeddings/transformer_large/transformer_large_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name transformer_large \
  --top-k 5 \
  --relevance-strategy tag_overlap
   ```

  CNN small
  ```
  python -m src.evaluation.run_batch_eval \
    data/wav \
    data/output/embeddings/cnn_small/cnn_small_embeddings.npy \
    data/output/embeddings/cnn_small/cnn_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy tag_overlap
  ```

  CNN medium
  ```
  python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/cnn_medium/cnn_medium_embeddings.npy \
  data/output/embeddings/cnn_medium/cnn_medium_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name cnn_medium \
  --top-k 5 \
  --relevance-strategy tag_overlap
  ```

  CNN Large
  ```
  python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/cnn_large/cnn_large_embeddings.npy \
  data/output/embeddings/cnn_large/cnn_large_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name cnn_large \
  --top-k 5 \
  --relevance-strategy tag_overlap
  ```
  ———

  # Notes

  - pseudo_labels.csv should contain track_id, wav_path, composer, and tag columns.
  - Retrieval/evaluation now prefers track_id matching over absolute path matching.
  - Both embedders are inference-only wrappers around pretrained models.

