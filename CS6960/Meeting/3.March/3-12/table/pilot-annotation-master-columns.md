# Pilot Annotation Master CSV
## Column Design for 20-Track Musical Character Tagging

This file explains the recommended column structure for `pilot_annotation_master.csv`.

---

## Core metadata columns

- `track_id`  
  Unique ID for each track.

- `title`  
  Track title.

- `composer`  
  Composer name.

- `file_path`  
  Local or project-relative audio file path.

- `duration_sec`  
  Duration in seconds.

- `instrumentation`  
  Short description such as `solo piano`, `string quartet`, `orchestra`.

---

## Music2Emo support columns

- `predicted_moods`  
  Semi-colon-separated mood predictions from Music2Emo.

- `valence`  
  Music2Emo valence score.

- `arousal`  
  Music2Emo arousal score.

These are support cues only, not final labels.

---

## Librosa support columns

- `spectral_centroid_mean`  
  Mean spectral centroid, used as a rough cue for brightness.

- `rms_mean`  
  Mean RMS energy, used as a rough cue for perceived energy or weight.

- `tempo_bpm`  
  Estimated tempo in beats per minute.

- `dynamic_range`  
  A summary statistic for loudness variation.

These are support cues only, not final labels.

---

## Human annotation columns

Fill these with:
- `1` if the tag applies
- `0` if the tag does not apply

Columns:
- `calm`
- `tense`
- `energetic`
- `lyrical`
- `bright`
- `heavy`

---

## Notes column

- `annotation_notes`  
  Free-text notes for uncertainty, repeated listening comments, or reasons for difficult decisions.

Example:
- `unclear between calm and lyrical`
- `strong low-register presence`
- `felt tense in opening only`

---

## Recommended workflow

1. Fill metadata first.
2. Import Music2Emo outputs.
3. Import Librosa features.
4. Listen to each track.
5. Assign binary human labels.
6. Add notes where needed.
7. After all 20 tracks are labeled, count tag frequency and refine the final tag set.

---

## Files included

- `pilot_annotation_master_template.csv`  
  Starter template with example mock rows.

