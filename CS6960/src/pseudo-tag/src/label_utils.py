"""Minimal helpers for tag labels and simple metadata extraction."""

from typing import Dict, List

import numpy as np


def tags_to_multihot(tag_dict: Dict[str, int], tag_list: List[str]) -> np.ndarray:
    """Convert a tag dictionary into a multi-hot vector."""
    return np.array([1 if tag_dict.get(tag, 0) else 0 for tag in tag_list], dtype=np.int64)


def compute_tag_overlap(tags_a: np.ndarray, tags_b: np.ndarray) -> int:
    """Count overlapping active tags between two binary vectors."""
    return int(np.sum((tags_a > 0) & (tags_b > 0)))


def compute_tag_jaccard(tags_a: np.ndarray, tags_b: np.ndarray) -> float:
    """Compute Jaccard similarity for two binary tag vectors."""
    intersection = np.sum((tags_a > 0) & (tags_b > 0))
    union = np.sum((tags_a > 0) | (tags_b > 0))
    if union == 0:
        return 0.0
    return float(intersection / union)


def extract_composer(track: Dict) -> str:
    """Return the composer name or 'unknown' if it is missing."""
    composer = track.get("composer", track.get("composers", "unknown"))
    if composer is None:
        return "unknown"

    composer_str = str(composer).strip()
    if not composer_str or composer_str.lower() == "nan":
        return "unknown"
    return composer_str


def to_binary_tag_value(value) -> int:
    """Convert strict binary-like values to 0/1."""
    if value is None:
        return 0

    if isinstance(value, bool):
        return int(value)

    val_str = str(value).strip().lower()

    if val_str in {"", "nan", "none"}:
        return 0
    if val_str in {"1", "1.0"}:
        return 1
    if val_str in {"0", "0.0"}:
        return 0

    raise ValueError(f"Non-binary tag value encountered: {value!r}")


def _is_binary_tag_column(rows: List[Dict], key: str) -> bool:
    """Check if a column behaves like a binary tag column (0/1)."""
    saw_explicit_binary = False

    for row in rows:
        val = row.get(key)

        if val is None:
            continue

        if isinstance(val, bool):
            saw_explicit_binary = True
            continue

        val_str = str(val).strip().lower()

        if val_str in {"", "nan", "none"}:
            continue
        if val_str in {"0", "0.0", "1", "1.0"}:
            saw_explicit_binary = True
            continue

        return False

    return saw_explicit_binary

def build_tag_list_from_columns(rows: List[Dict]) -> List[str]:
    """Detect tag column names from binary tag fields in the rows."""
    if not rows:
        return []

    excluded = {"track_id", "valence", "arousal"}
    all_keys = set()

    for row in rows:
        all_keys.update(row.keys())

    tag_names = []

    for key in sorted(all_keys):
        if key in excluded:
            continue

        if _is_binary_tag_column(rows, key):
            tag_names.append(key)

    return tag_names


def row_to_tag_dict(row: Dict, tag_list: List[str]) -> Dict[str, int]:
    """Extract binary tag values safely from a row."""
    return {tag: to_binary_tag_value(row.get(tag)) for tag in tag_list}


def row_to_multihot(row: Dict, tag_list: List[str]) -> np.ndarray:
    """Convert a row with tag columns into a multi-hot vector."""
    tag_dict = row_to_tag_dict(row, tag_list)
    return tags_to_multihot(tag_dict, tag_list)
