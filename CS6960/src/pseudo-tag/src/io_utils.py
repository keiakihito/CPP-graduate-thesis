"""Minimal file I/O helpers for JSON and CSV."""

import csv
import json
from pathlib import Path
from typing import Dict, List


def ensure_dir(path: Path) -> None:
    """Create parent directories for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict | list:
    """Load JSON data from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    """Save JSON data to disk."""
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv(path: Path) -> List[Dict[str, str]]:
    """Load CSV rows as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: List[Dict], path: Path) -> None:
    """Save dictionary rows to a CSV file."""
    ensure_dir(path)
    fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
