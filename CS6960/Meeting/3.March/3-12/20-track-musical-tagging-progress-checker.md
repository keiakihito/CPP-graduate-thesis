# 20-Track Musical Character Tagging Workflow
## Step-by-Step Progress Checker for Pilot Annotation

This checklist integrates the **Music2Emo-assisted cue generation** and **Librosa-based feature support** workflow into a single pilot annotation process for 20 tracks.

---

## Objective

Create a small, defensible pilot annotation set for **musical character retrieval** by:

- starting from a **candidate tag set**
- using **Music2Emo** and **Librosa** only as **annotation support tools**
- assigning final labels through **human listening**
- refining the tag set based on pilot consistency
- preparing the final scheme for **full-archive annotation**

---

## Initial Candidate Tags

Start with the following six candidate tags:

- calm
- tense
- energetic
- lyrical
- bright
- heavy

These were chosen because they combine:
- emotion-oriented descriptors commonly used in music emotion research
- perceptually meaningful descriptors that may be useful in classical-music retrieval
- a manageable starting vocabulary for pilot annotation

---

## Ground Rule

**Do not use Music2Emo or Librosa outputs as ground-truth labels.**  
They are used only as **supporting cues** during annotation.

Final labels must be assigned by **human listening** using consistent tag definitions.

---

# Phase 1 — Prepare the Pilot Set

## Step 1. Select 20 pilot tracks
Choose 20 tracks that cover a wide range of musical character.

### Progress checker
- [ ] Selected 20 total tracks
- [ ] Included calm / low-intensity examples
- [ ] Included tense / dramatic examples
- [ ] Included energetic / high-motion examples
- [ ] Included lyrical examples
- [ ] Included different instrumentation types
- [ ] Included solo piano
- [ ] Included chamber music
- [ ] Included orchestral works
- [ ] Saved the final pilot track list

### Notes
Try to maximize variety rather than random sampling alone.

---

## Step 2. Create the pilot metadata sheet
Create a CSV or spreadsheet with one row per track.

### Minimum columns
- track_id
- title
- composer
- file_path
- duration
- instrumentation
- notes

### Progress checker
- [ ] Created pilot metadata file
- [ ] Assigned unique track IDs
- [ ] Confirmed all file paths are correct
- [ ] Confirmed all 20 tracks are accessible

---

# Phase 2 — Generate Annotation Support Cues

## Step 3. Run Music2Emo on all 20 tracks
For each track, extract:
- predicted_moods
- valence
- arousal

### Example mock output
```text
track_001.mp3
predicted_moods = ["tense", "dramatic"]
valence = 4.2
arousal = 6.8

track_002.mp3
predicted_moods = ["calm", "sad", "melancholic"]
valence = 3.0
arousal = 2.5

track_003.mp3
predicted_moods = ["energetic", "epic", "dramatic"]
valence = 6.0
arousal = 8.7
```

### Progress checker
- [ ] Ran Music2Emo on all 20 tracks
- [ ] Saved predicted_moods for all tracks
- [ ] Saved valence scores for all tracks
- [ ] Saved arousal scores for all tracks
- [ ] Exported Music2Emo results to CSV

---

## Step 4. Run Librosa feature extraction on all 20 tracks
Extract low-level support features such as:
- spectral centroid
- RMS energy
- tempo
- dynamic range

### Example mock output
```csv
track_id,title,valence,arousal,spectral_centroid,rms,tempo,dynamic_range
001,Beethoven_String_Quartet_1,4.2,6.8,1850,0.09,126,0.21
002,Chopin_Nocturne_Op9_No2,3.0,2.5,980,0.04,72,0.10
003,Tchaikovsky_1812_Overture,6.0,8.7,2400,0.18,138,0.35
```

### Progress checker
- [ ] Extracted spectral centroid for all tracks
- [ ] Extracted RMS for all tracks
- [ ] Extracted tempo for all tracks
- [ ] Extracted dynamic range for all tracks
- [ ] Exported Librosa features to CSV

---

## Step 5. Merge support cues into one annotation sheet
Combine metadata, Music2Emo outputs, and Librosa outputs into one pilot annotation table.

### Recommended columns
- track_id
- title
- composer
- predicted_moods
- valence
- arousal
- spectral_centroid
- rms
- tempo
- dynamic_range
- calm
- tense
- energetic
- lyrical
- bright
- heavy
- annotation_notes

### Progress checker
- [ ] Merged metadata with Music2Emo output
- [ ] Merged metadata with Librosa features
- [ ] Added binary tag columns
- [ ] Added annotation_notes column
- [ ] Saved the master pilot annotation sheet

---

# Phase 3 — Human Annotation

## Step 6. Define the operational meaning of each tag
Before listening, write one short definition for each tag.

### Example operational definitions
- **calm**: low tension, low agitation, and a stable expressive impression
- **tense**: urgency, instability, or unresolved pressure is perceptually salient
- **energetic**: strong activity, forward motion, or driving force is prominent
- **lyrical**: melody feels flowing, expressive, and song-like
- **bright**: timbre feels clear, brilliant, or high-frequency prominent
- **heavy**: music conveys density, weight, or strong low-register presence

### Progress checker
- [ ] Wrote a definition for calm
- [ ] Wrote a definition for tense
- [ ] Wrote a definition for energetic
- [ ] Wrote a definition for lyrical
- [ ] Wrote a definition for bright
- [ ] Wrote a definition for heavy
- [ ] Confirmed definitions are short and consistent

---

## Step 7. Listen and assign binary labels
Listen to each track and assign:
- 1 = tag applies
- 0 = tag does not apply

Use Music2Emo and Librosa only as supporting cues, not as automatic decisions.

### Progress checker
- [ ] Annotated tracks 1–5
- [ ] Annotated tracks 6–10
- [ ] Annotated tracks 11–15
- [ ] Annotated tracks 16–20
- [ ] Added notes where tag assignment was uncertain
- [ ] Reviewed all 20 rows for consistency

### Annotation rule
If a tag feels unclear or unstable across repeated listening, make a note in `annotation_notes`.

---

# Phase 4 — Tag Refinement

## Step 8. Count tag frequency across the 20 tracks
After annotation, count how often each tag was used.

### Example frequency table
| tag | frequency |
|---|---:|
| calm | 12 |
| tense | 9 |
| energetic | 7 |
| lyrical | 14 |
| bright | 1 |
| heavy | 2 |

### Progress checker
- [ ] Counted tag frequency
- [ ] Identified very rare tags
- [ ] Identified ambiguous tags
- [ ] Reviewed notes for unstable tags

---

## Step 9. Remove weak or inconsistent tags
Remove tags that are:
- rarely applicable
- difficult to interpret consistently
- not useful in the classical archive context

### Example refinement
If **bright** and **heavy** are rarely used or too unstable, refine the final set to:

- calm
- tense
- energetic
- lyrical

### Progress checker
- [ ] Selected final refined tag set
- [ ] Removed weak tags
- [ ] Documented why removed tags were excluded
- [ ] Confirmed final tag set is small and stable

---

# Phase 5 — Prepare for Full Annotation

## Step 10. Freeze the final tag definitions
Once the final set is chosen, do not change the meaning of the tags.

### Progress checker
- [ ] Final tag list is fixed
- [ ] Final tag definitions are fixed
- [ ] Annotation rules are fixed
- [ ] Saved the final labeling guideline

---

## Step 11. Apply the refined tag set to the full archive
Use the refined final tags to label all tracks in the archive.

### Progress checker
- [ ] Created full-archive annotation sheet
- [ ] Added final tag columns only
- [ ] Started full annotation
- [ ] Completed 25% of archive
- [ ] Completed 50% of archive
- [ ] Completed 75% of archive
- [ ] Completed 100% of archive
- [ ] Exported final annotation CSV

---

# Suggested Deliverables

By the end of this workflow, you should have:

- [ ] `pilot_track_list.csv`
- [ ] `music2emo_outputs.csv`
- [ ] `librosa_features.csv`
- [ ] `pilot_annotation_master.csv`
- [ ] `tag_frequency_summary.csv`
- [ ] `final_tag_guideline.md`
- [ ] `full_archive_annotation.csv`

---

# Recommended Thesis Wording

You may describe the process like this:

> A small set of candidate musical character tags was first defined based on perceptual descriptors commonly used in music emotion research. A pilot annotation was then conducted on 20 tracks to evaluate whether these tags could be applied consistently within the classical music archive. During this pilot phase, Music2Emo predictions and low-level audio features extracted with Librosa were used only as supporting cues. Final labels were assigned through human listening according to predefined operational definitions. Tags that were rarely applicable or difficult to apply consistently were removed, and the refined tag set was then applied to the full archive.

---

# Quick Summary

## Final workflow
- [ ] Define 6 candidate tags
- [ ] Select 20 varied pilot tracks
- [ ] Run Music2Emo
- [ ] Run Librosa
- [ ] Merge all support information
- [ ] Listen and assign human labels
- [ ] Count tag usage
- [ ] Refine to the final 4-ish tags
- [ ] Freeze definitions
- [ ] Annotate the full archive
