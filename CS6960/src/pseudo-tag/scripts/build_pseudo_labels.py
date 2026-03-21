import os
import json
import csv
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3


# =========================
# Config
# =========================
BUCKET_NAME = os.getenv("IPALPITI_S3_BUCKET", "ipalpiti-audio-resource")
METADATA_PATH = Path("metadata/tracks.json")
RAW_DIR = Path("data/raw")
WAV_DIR = Path("data/wav")
OUTPUT_CSV = Path("data/output/pseudo_labels.csv")

# Optional: limit for MVP test
MAX_TRACKS = 3  # change to 20 later


# =========================
# Utility
# =========================
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
    """
    Replace this with your actual Music2Emo inference call.

    Expected return:
    {
        "valence": 0.0 ~ 1.0,
        "arousal": 0.0 ~ 1.0
    }
    """
    return {
        "valence": 0.5,
        "arousal": 0.5
    }

# =========================
# Pseudo tags
# =========================
def derive_pseudo_tags(valence, arousal):
    return {
        "energetic": int(valence >= 0.6 and arousal >= 0.6),
        "tense":     int(valence <  0.4 and arousal >= 0.6),
        "calm":      int(valence >= 0.6 and arousal <  0.4),
        "lyrical":   int(0.3 <= arousal <= 0.6 and 0.4 <= valence <= 0.7),
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

    tags = derive_pseudo_tags(valence, arousal)

    result = {
        "track_id": track_id,
        "title": title,
        "album_name": track.get("album_name"),
        "composer": track.get("composers"),
        "audio_s3_key": s3_key,
        "wav_path": str(wav_path),
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "energetic": tags["energetic"],
        "tense": tags["tense"],
        "calm": tags["calm"],
        "lyrical": tags["lyrical"],
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
    ensure_dirs()
    tracks = load_tracks(METADATA_PATH)
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

    save_results_csv(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()