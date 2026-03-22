"""Build binary relevance labels for retrieval evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io_utils import load_csv, load_json
from src.label_utils import (
    build_tag_list_from_columns,
    compute_tag_overlap,
    extract_composer,
    row_to_multihot,
)


def load_label_data(path: str) -> list[dict[str, Any]]:
    """Load label rows from a JSON or CSV file."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        data = load_json(file_path)
        if not isinstance(data, list):
            raise ValueError("JSON label data must be a list of dictionaries.")
        return data

    if suffix == ".csv":
        return load_csv(file_path)

    raise ValueError(f"Unsupported label data format: {path}")


def build_label_index(label_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a simple lookup by file_id and path."""
    index: dict[str, dict[str, Any]] = {}

    for row in label_data:
        if not isinstance(row, dict):
            continue

        if "track_id" in row and row["track_id"] is not None:
            index[str(row["track_id"])] = row
        if "file_id" in row and row["file_id"] is not None:
            index[str(row["file_id"])] = row
        if "path" in row and row["path"] is not None:
            index[str(row["path"])] = row
        if "wav_path" in row and row["wav_path"] is not None:
            index[str(row["wav_path"])] = row

    return index


def extract_track_id(value: str) -> str | None:
    """Extract a track id from names like 'track_30_name.wav'."""
    stem = Path(str(value)).stem
    parts = stem.split("_")

    if len(parts) >= 2 and parts[0].lower() == "track" and parts[1].isdigit():
        return parts[1]

    return None


def same_composer_relevance(query_row: dict[str, Any], candidate_row: dict[str, Any]) -> int:
    """Return 1 when query and candidate share the same known composer."""
    query_composer = extract_composer(query_row)
    candidate_composer = extract_composer(candidate_row)

    if query_composer == "unknown" or candidate_composer == "unknown":
        return 0

    return int(query_composer == candidate_composer)

def shared_tag_overlap_relevance(
    query_row: dict[str, Any],
    candidate_row: dict[str, Any],
    tag_list: list[str],
) -> int:
    """Return 1 when query and candidate share any active tag."""
    query_tags = row_to_multihot(query_row, tag_list)
    candidate_tags = row_to_multihot(candidate_row, tag_list)
    return int(compute_tag_overlap(query_tags, candidate_tags) > 0)


def _get_item_key(item: dict[str, Any]) -> str | None:
    """Extract a lookup key, preferring stable track_id when possible."""
    track_id = item.get("track_id")
    if track_id is not None:
        return str(track_id)

    if "file_id" in item and item["file_id"] is not None:
        extracted = extract_track_id(str(item["file_id"]))
        if extracted is not None:
            return extracted
        return str(item["file_id"])
    if "path" in item and item["path"] is not None:
        extracted = extract_track_id(str(item["path"]))
        if extracted is not None:
            return extracted
        return str(item["path"])
    return None


def build_relevance(
    query_key: str,
    retrieved_items: list[dict[str, Any]],
    label_data: list[dict[str, Any]],
    strategy: str = "composer",
) -> list[int]:
    """Build a binary relevance list aligned with retrieved items."""
    label_index = build_label_index(label_data)
    query_row = label_index.get(str(query_key))

    if query_row is None:
        raise ValueError(f"Query id/path not found in label data: {query_key}")

    tag_list = build_tag_list_from_columns(label_data) if strategy == "tag_overlap" else []
    relevance: list[int] = []

    for item in retrieved_items:
        item_key = _get_item_key(item)
        candidate_row = label_index.get(item_key) if item_key is not None else None

        if candidate_row is None:
            relevance.append(0)
            continue

        if strategy == "composer":
            relevance.append(same_composer_relevance(query_row, candidate_row))
        elif strategy == "tag_overlap":
            relevance.append(shared_tag_overlap_relevance(query_row, candidate_row, tag_list))
        else:
            raise ValueError(f"Unsupported relevance strategy: {strategy}")

    return relevance
