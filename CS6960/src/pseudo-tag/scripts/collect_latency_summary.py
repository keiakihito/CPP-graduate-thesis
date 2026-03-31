"""Run embedding generation for all models and collect latency results."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


LATENCY_PATTERN = re.compile(r"Average latency per track:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = repo_root / "data" / "output" / "latency_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    commands = {
        "cnn_small": [
            sys.executable,
            "-m",
            "src.pipeline.build_embeddings",
            "data/wav",
            "data/output/embeddings/cnn_small",
            "--model-name",
            "cnn_small",
        ],
        "cnn_medium": [
            sys.executable,
            "-m",
            "src.pipeline.build_embeddings",
            "data/wav",
            "data/output/embeddings/cnn_medium",
            "--model-name",
            "cnn_medium",
        ],
        "cnn_large": [
            sys.executable,
            "-m",
            "src.pipeline.build_embeddings",
            "data/wav",
            "data/output/embeddings/cnn_large",
            "--model-name",
            "cnn_large",
        ],
        "transformer_medium": [
            sys.executable,
            "-m",
            "src.pipeline.build_embeddings",
            "data/wav",
            "data/output/embeddings/transformer_medium",
            "--model-name",
            "transformer_medium",
        ],
        "transformer_large": [
            sys.executable,
            "-m",
            "src.pipeline.build_embeddings",
            "data/wav",
            "data/output/embeddings/transformer_large",
            "--model-name",
            "transformer_large",
        ],
    }

    results: dict[str, str | None] = {}

    for model_name, command in commands.items():
        print(f"Running {model_name}...")
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        stdout = completed.stdout
        stderr = completed.stderr

        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)

        match = LATENCY_PATTERN.search(stdout)
        if match:
            results[model_name] = match.group(1)
            print(f"Captured latency for {model_name}: {results[model_name]} ms")
        else:
            results[model_name] = None
            print(f"Latency output not found for {model_name}.")

    lines = [
        "Model | Latency (ms)",
        "-------------------",
    ]
    for model_name in commands:
        latency = results[model_name] if results[model_name] is not None else "N/A"
        lines.append(f"{model_name} | {latency}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote latency summary to {summary_path}")


if __name__ == "__main__":
    main()
