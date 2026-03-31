#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

WAV_DIR="data/wav"
LABELS_PATH="data/output/labels/pseudo_labels.csv"
OUTPUT_DIR="data/output/evaluation"

mkdir -p "$OUTPUT_DIR"

PROXY1_OUT="$OUTPUT_DIR/proxy_task_1_composer.txt"
PROXY2_OUT="$OUTPUT_DIR/proxy_task_2_tag_overlap.txt"
SUMMARY_OUT="$OUTPUT_DIR/summary.txt"

MODELS=(
  "transformer_medium"
  "transformer_large"
  "cnn_small"
  "cnn_medium"
  "cnn_large"
)

run_eval() {
  local model_name="$1"
  local relevance_strategy="$2"
  local output_file="$3"
  local embeddings_dir="data/output/embeddings/${model_name}"
  local embeddings_path="${embeddings_dir}/${model_name}_embeddings.npy"
  local metadata_path="${embeddings_dir}/${model_name}_metadata.json"

  {
    echo "============================================================"
    echo "model_name: ${model_name}"
    echo "relevance_strategy: ${relevance_strategy}"
    echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"
  } >> "$output_file"

  "$PYTHON_BIN" -m src.evaluation.run_batch_eval \
    "$WAV_DIR" \
    "$embeddings_path" \
    "$metadata_path" \
    "$LABELS_PATH" \
    --model-name "$model_name" \
    --top-k 5 \
    --relevance-strategy "$relevance_strategy" >> "$output_file" 2>&1

  echo "" >> "$output_file"
}

write_summary() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

proxy_files = [
    ("Proxy Task 1", Path("data/output/evaluation/proxy_task_1_composer.txt")),
    ("Proxy Task 2", Path("data/output/evaluation/proxy_task_2_tag_overlap.txt")),
]
summary_path = Path("data/output/evaluation/summary.txt")

def extract_summary_blocks(path: Path) -> list[str]:
    blocks = []
    lines = path.read_text().splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("model_name: "):
            model_name = line.split(": ", 1)[1]
            relevance_strategy = ""
            date = ""
            summary_values: dict[str, str] = {}

            j = i + 1
            while j < len(lines):
                current = lines[j].strip()
                if current.startswith("relevance_strategy: "):
                    relevance_strategy = current.split(": ", 1)[1]
                elif current.startswith("date: "):
                    date = current.split(": ", 1)[1]
                elif current == "=== Batch Evaluation Summary ===":
                    k = j + 1
                    while k < len(lines):
                        entry = lines[k].strip()
                        if not entry:
                            break
                        if entry.startswith("==="):
                            break
                        if entry.startswith("model_name: "):
                            break
                        if ": " in entry:
                            key, value = entry.split(": ", 1)
                            summary_values[key] = value
                        k += 1
                    blocks.append(
                        "\n".join(
                            [
                                f"model_name: {model_name}",
                                f"relevance_strategy: {relevance_strategy}",
                                f"date: {date}",
                                f"total_queries: {summary_values.get('total_queries', '')}",
                                f"evaluated_queries: {summary_values.get('evaluated_queries', '')}",
                                f"missing_label_queries: {summary_values.get('missing_label_queries', '')}",
                                f"queries_with_hit@5: {summary_values.get('queries_with_hit@5', '')}",
                                f"precision@5: {summary_values.get('precision@5', '')}",
                                f"recall@5: {summary_values.get('recall@5', '')}",
                                f"f1@5: {summary_values.get('f1@5', '')}",
                                f"ndcg@5: {summary_values.get('ndcg@5', '')}",
                                f"hit@5: {summary_values.get('hit@5', '')}",
                            ]
                        )
                    )
                    i = k
                    break
                j += 1
        i += 1

    return blocks

with summary_path.open("w", encoding="utf-8") as f:
    for title, path in proxy_files:
        f.write(f"# {title}\n")
        for block in extract_summary_blocks(path):
            f.write(block)
            f.write("\n\n")
PY
}

: > "$PROXY1_OUT"
: > "$PROXY2_OUT"

for model_name in "${MODELS[@]}"; do
  run_eval "$model_name" "composer" "$PROXY1_OUT"
done

for model_name in "${MODELS[@]}"; do
  run_eval "$model_name" "tag_overlap" "$PROXY2_OUT"
done

write_summary

echo "Saved Proxy Task 1 results to: $PROXY1_OUT"
echo "Saved Proxy Task 2 results to: $PROXY2_OUT"
echo "Saved summary to: $SUMMARY_OUT"
