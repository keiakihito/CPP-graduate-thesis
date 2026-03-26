import os
import sys
import json
import csv
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import unicodedata

import boto3
import numpy as np

MUSIC2EMO_DIR = Path(__file__).resolve().parents[1] / "Music2Emotion"
sys.path.insert(0, str(MUSIC2EMO_DIR))
from music2emo import Music2emo

# =========================
# Config
# =========================
BUCKET_NAME = os.getenv("IPALPITI_S3_BUCKET", "ipalpiti-audio-resource")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "metadata" / "tracks.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
WAV_DIR = PROJECT_ROOT / "data" / "wav"
OUTPUT_CSV = PROJECT_ROOT / "data" / "output" / "pseudo_labels.csv"

# Optional: limit for MVP test. Set to None to process the full dataset.
MAX_TRACKS = None
LOW_QUANTILE = 0.33
HIGH_QUANTILE = 0.67
MIDDLE_LOW_QUANTILE = 0.40
MIDDLE_HIGH_QUANTILE = 0.60

def load_music2emo():
    old_cwd = os.getcwd()
    try:
        os.chdir(MUSIC2EMO_DIR)
        model = Music2emo()
    finally:
        os.chdir(old_cwd)
    return model

music2emo_model = None

# =========================
# Utility
# =========================
def normalize_1to9(x: float) -> float:
    return (x - 1.0) / 8.0


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_tracks(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tracks.json must be a JSON array")
    return data


def safe_stem(track_id: Any, title: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)
    cleaned = cleaned[:80].strip("_")
    return f"track_{track_id}_{cleaned}"


def normalize_composer_name(composer: Any) -> str:
    """Normalize composer names into a stable evaluation field."""
    if composer is None:
        return "unknown"

    composer_str = str(composer).strip()
    if not composer_str or composer_str.lower() == "nan":
        return "unknown"

    normalized = unicodedata.normalize("NFKD", composer_str)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[\.\-_,/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    aliases = {
        "george enesco": "george enescu",
        "georges enesco": "george enescu",
        "georges enescu": "george enescu",
        "edvard greig": "edvard grieg",
        "frederic chopin": "frederic chopin",
        "w a mozart": "wolfgang amadeus mozart",
        "mozart": "wolfgang amadeus mozart",
        "j s bach": "johann sebastian bach",
        "bach": "johann sebastian bach",
        "piotr tchaikovsky": "pyotr ilyich tchaikovsky",
        "p i tchaikovsky": "pyotr ilyich tchaikovsky",
        "peter i tchaikovsky": "pyotr ilyich tchaikovsky",
        "tchaikovsky": "pyotr ilyich tchaikovsky",
        "ludvig van beethoven": "ludwig van beethoven",
        "ludwig van beethoven g mahler": "ludwig van beethoven gustav mahler",
        "handel lerman": "handel lerman",
        "handel halvorsen": "handel halvorsen",
        "albinoni": "tomaso albinoni",
        "tomaso vitali": "tommaso antonio vitali",
        "vivaldi": "antonio vivaldi",
    }

    return aliases.get(normalized, normalized)



# =========================
# S3 download
# =========================
def download_audio_from_s3(s3_client, s3_key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(BUCKET_NAME, s3_key, str(local_path))


# =========================
# Audio conversion
# =========================
def convert_to_wav(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",          # mono
        "-ar",
        "16000",      # 16kHz
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# =========================
# Music2Emo inference
# =========================
def predict_valence_arousal(wav_path: Path) -> Dict[str, float]:
    old_cwd = os.getcwd()
    try:
        os.chdir(MUSIC2EMO_DIR)
        output_dic = music2emo_model.predict(str(wav_path.resolve()))
    finally:
        os.chdir(old_cwd)

    raw_valence = float(output_dic["valence"])
    raw_arousal = float(output_dic["arousal"])

    return {
        "valence": normalize_1to9(raw_valence),
        "arousal": normalize_1to9(raw_arousal),
    }


# =========================
# Pseudo tags
# =========================
def compute_va_thresholds(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute dataset-relative valence/arousal thresholds from all rows."""
    if not rows:
        raise ValueError("Cannot compute thresholds from empty rows.")

    valences = np.array([float(row["valence"]) for row in rows], dtype=float)
    arousals = np.array([float(row["arousal"]) for row in rows], dtype=float)

    return {
        "valence_low": float(np.quantile(valences, LOW_QUANTILE)),
        "valence_high": float(np.quantile(valences, HIGH_QUANTILE)),
        "valence_middle_low": float(np.quantile(valences, MIDDLE_LOW_QUANTILE)),
        "valence_middle_high": float(np.quantile(valences, MIDDLE_HIGH_QUANTILE)),
        "arousal_low": float(np.quantile(arousals, LOW_QUANTILE)),
        "arousal_high": float(np.quantile(arousals, HIGH_QUANTILE)),
        "arousal_middle_low": float(np.quantile(arousals, MIDDLE_LOW_QUANTILE)),
        "arousal_middle_high": float(np.quantile(arousals, MIDDLE_HIGH_QUANTILE)),
    }


def derive_pseudo_tags(
    valence: float,
    arousal: float,
    thresholds: Dict[str, float],
) -> Dict[str, int]:
    """Derive binary pseudo tags from dataset-relative VA thresholds.

    Quantile-based thresholds are used because predicted valence/arousal values
    are concentrated in a narrow mid-range in this dataset.
    """
    valence_low = thresholds["valence_low"]
    valence_high = thresholds["valence_high"]
    arousal_low = thresholds["arousal_low"]
    arousal_high = thresholds["arousal_high"]

    is_middle_valence = (
        thresholds["valence_middle_low"] < valence < thresholds["valence_middle_high"]
    )
    is_middle_arousal = (
        thresholds["arousal_middle_low"] < arousal < thresholds["arousal_middle_high"]
    )

    # Axis-based tags are denser than quadrant-based tags for this dataset,
    # where valence/arousal predictions cluster in a narrow central range.
    return {
        "energetic": int(arousal >= arousal_high),
        "tense": int(valence <= valence_low),
        "calm": int(arousal <= arousal_low),
        "lyrical": int(is_middle_valence and is_middle_arousal),
    }


# =========================
# Main pipeline
# =========================
def process_track(s3_client, track: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    track_id = track.get("track_id")
    title = track.get("title", "untitled")
    s3_key = track.get("audio_s3_key")

    if not s3_key:
        print(f"[SKIP] track_id={track_id}: missing audio_s3_key")
        return None

    stem = safe_stem(track_id, title)
    raw_ext = Path(s3_key).suffix or ".audio"
    raw_path = RAW_DIR / f"{stem}{raw_ext}"
    wav_path = WAV_DIR / f"{stem}.wav"

    print(f"[INFO] Downloading: {s3_key}")
    download_audio_from_s3(s3_client, s3_key, raw_path)

    print(f"[INFO] Converting to wav: {raw_path.name}")
    convert_to_wav(raw_path, wav_path)

    print(f"[INFO] Running Music2Emo: {wav_path.name}")
    pred = predict_valence_arousal(wav_path)
    valence = float(pred["valence"])
    arousal = float(pred["arousal"])

    result = {
        "track_id": track_id,
        "title": title,
        "album_name": track.get("album_name"),
        "composer": track.get("composers"),
        "composer_eval": normalize_composer_name(track.get("composers")),
        "audio_s3_key": s3_key,
        "wav_path": str(wav_path),
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "review_flag": "",
        "review_corrected_tags": "",
        "review_notes": "",
    }
    return result


def save_results_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        print("[WARN] No rows to save.")
        return

    fieldnames = [
        "track_id",
        "title",
        "album_name",
        "composer",
        "composer_eval",
        "audio_s3_key",
        "wav_path",
        "valence",
        "arousal",
        "energetic",
        "tense",
        "calm",
        "lyrical",
        "review_flag",
        "review_corrected_tags",
        "review_notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Saved CSV: {output_path}")


def main() -> None:
    global music2emo_model
    music2emo_model = load_music2emo()

    ensure_dirs()
    tracks = load_tracks(METADATA_PATH)
    if MAX_TRACKS is not None:
        tracks = tracks[:MAX_TRACKS]

    s3_client = boto3.client("s3")
    results: List[Dict[str, Any]] = []

    for track in tracks:
        try:
            row = process_track(s3_client, track)
            if row is not None:
                results.append(row)
        except Exception as e:
            print(f"[ERROR] track_id={track.get('track_id')}: {e}")

    thresholds = compute_va_thresholds(results)
    print(
        "[INFO] VA thresholds: "
        f"valence_low={thresholds['valence_low']:.4f}, "
        f"valence_high={thresholds['valence_high']:.4f}, "
        f"arousal_low={thresholds['arousal_low']:.4f}, "
        f"arousal_high={thresholds['arousal_high']:.4f}"
    )

    for row in results:
        tags = derive_pseudo_tags(float(row["valence"]), float(row["arousal"]), thresholds)
        row["energetic"] = tags["energetic"]
        row["tense"] = tags["tense"]
        row["calm"] = tags["calm"]
        row["lyrical"] = tags["lyrical"]

    save_results_csv(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
