# Use‑Case Catalog (No Authentication)

This document describes the functional use cases for the **Audio Segmentation MVP** shown in the wireframe: a visitor browses albums, opens a track, selects a time range on the waveform, and downloads the trimmed clip. Authentication is **out of scope** for this phase, so all flows assume an anonymous visitor. The backend is assumed to be AWS-based (API Gateway → Lambda → RDS/S3) with presigned S3 URLs for downloads.

---

## Actors

**Primary**
- **Visitor** — An anonymous user interacting via a web browser. They can browse the catalog, preview tracks, define a segment, and download a trimmed clip.

**Supporting**
- **System** — The combination of Web UI + API endpoints + Lambda functions that handle requests and orchestrate processing.
- **Storage** — **RDS** for album/track metadata and **S3** for original audio, waveform assets, and temporary trimmed clips.

---

## UC‑01: Browse Albums

**Goal.** Let the Visitor discover available content by viewing a list of albums and opening one to see its tracks.

**Preconditions.**
- The catalog has been imported and contains one or more albums.
- The system can read from RDS and render the Albums list.

**Trigger.** The Visitor lands on the site and clicks **Albums**.

**Main flow.**
- Show a paginated grid/list of albums with basic metadata (title, artist, optional year/cover).
- Each album item is clickable, leading to the album’s Tracks page.
- The list should render quickly and handle slow networks gracefully (skeletal loading states).

**Alternate/Exceptions.**
- If there are **no albums**, show an empty state with a friendly message and a **Retry**/refresh action.
- If RDS cannot be reached, show a non-blocking error toast and allow the Visitor to retry.

**Postconditions.**
- None beyond navigation; the Visitor has context and can proceed to a chosen album.

**Notes.**
- Pagination is preferred over infinite scroll for predictable performance.
- Optional query params (`?page=n`) should be bookmarkable.

---

## UC‑02: View Tracks in Album

**Goal.** Show all tracks for a selected album so the Visitor can pick one to preview and segment.

**Preconditions.**
- The album exists in RDS and is addressable by `albumId`.
- Tracks for the album have been imported and linked.

**Trigger.** The Visitor clicks an album from the Albums list or follows a deep link to the album page.

**Main flow.**
- Display a list of tracks with title and duration (and track number if available).
- Each track is selectable; clicking opens the **Track page**.
- Provide stable URLs so the page is shareable/bookmarkable.

**Alternate/Exceptions.**
- If the album is not found, return a **404** page with a link back to Albums.
- If the album has **no tracks**, show an empty state but keep navigation intact.

**Postconditions.**
- The Visitor can proceed to a specific track for preview and segmentation.

**Notes.**
- For very large albums, paginate or virtualize the list to keep the UI responsive.

---

## UC‑03: Preview a Track

**Goal.** Allow the Visitor to listen to a track, see its waveform, and control playback (Play/Stop/Seek).

**Preconditions.**
- The track exists and has a valid audio source in S3.
- Waveform data (JSON or image) is accessible, or the UI can generate it on the fly.

**Trigger.** The Visitor opens a track page.

**Main flow.**
- Render the waveform and standard controls: **Play**, **Stop**, **–5s**, **+5s**.
- Play starts from the current playhead; seek buttons adjust the playhead while continuing playback.
- Keep the UI responsive while audio buffers; show progress or a spinner if needed.
- (Optional, but recommended) Provide keyboard shortcuts: **Space** (play/pause), **←/→** (±5s).

**Alternate/Exceptions.**
- If the audio cannot be played (format or network), show an actionable error and a **Retry** button.
- If waveform data is missing, fall back to a basic scrubber control.

**Postconditions.**
- The Visitor hears the track and can position the playhead where needed.

**Notes.**
- Keep latency low for a tight “scrub/play” feel (pre-buffer a small window around the playhead).

---

## UC‑04: Define Segment

**Goal.** Enable the Visitor to select the desired **start** and **end** of a clip using draggable handles (“selection box”).

**Preconditions.**
- The track page has loaded with a visible waveform and current playhead position.
- The UI knows the track duration (`durationMs`).

**Trigger.** The Visitor clicks/drag on the waveform or grabs the selection handles.

**Main flow.**
- Provide **left** and **right** handles that define `startMs` and `endMs`.
- Snap the playhead to the **left handle** when the Visitor presses **Play**, so playback starts at the selected start.
- Update the visible timecodes live while dragging; allow fine-grained adjustments (e.g., via arrow keys).
- Validate the selection continuously: **startMs < endMs**, and within configured min/max duration.

**Alternate/Exceptions.**
- If **start ≥ end**, disable **Download** and show a tooltip (“Start must be before End”).
- If the selection is **too short/long**, show a non-intrusive hint and keep the handles in place.

**Postconditions.**
- A valid segment is selected and ready to be downloaded (or the UI clearly explains why it isn’t).

**Notes.**
- Consider snapping to zero-crossings or frames for clean cuts (can be deferred).
- Expose time in both mm:ss.sss and absolute milliseconds for precision work.

---

## UC‑05: Download Segment

**Goal.** Produce and deliver a trimmed audio file for the currently selected time range.

**Preconditions.**
- A valid segment is selected (`startMs < endMs`) and satisfies min/max constraints.
- Backend has access to the source audio in S3 and can execute a trim job (e.g., via ffmpeg in Lambda).

**Trigger.** The Visitor clicks **Download**.

**Main flow.**
- The UI sends a request: `{ trackId, startMs, endMs, format? }` to the API.
- The API validates inputs (track existence, range within duration, clip length within limits).
- On success, the API invokes the **Segmentation Lambda**, which reads from S3, trims/encodes, stores the result as a **temporary object**, and returns a **presigned URL** and filename.
- The API forwards the URL to the UI; the browser initiates the download immediately.
- Temporary clips are managed with an S3 **lifecycle** policy (e.g., 24h expiry).

**Alternate/Exceptions.**
- **400** invalid range or missing params → UI shows an inline error and keeps the current selection.
- **413** clip too long → UI shows message with max duration and a quick-fix suggestion.
- **429** rate-limited (anonymous use) → back off and retry after the provided window.
- **5xx** processing error → show a toast (“Something went wrong”) with **Retry**.

**Postconditions.**
- The Visitor has a local audio file of the trimmed segment; the presigned URL will expire per TTL.

**Notes.**
- Choose a default output format now (e.g., **WAV** 16‑bit PCM or **MP3** 192kbps) to keep the UI simple.
- Filenames should be deterministic: `Album_Track_startMs-endMs.ext`.

---

## Future Extensions (for later phases)

**UC‑06: Search & Filter**  
- Purpose: Quickly locate albums or tracks.  
- Scope: Client-side filtering for small catalogs; API search for large datasets.

**UC‑07: Catalog Import (Back‑office)**  
- Purpose: Keep RDS in sync from `albums.csv` and `tracks.csv`.  
- Scope: Scheduled Lambda job, idempotent upserts, metrics/logging dashboard.

**UC‑08: Waveform Generation & Cache**  
- Purpose: Precompute and cache waveform data to speed up track loads.  
- Scope: Batch processor stores reduced-resolution JSON/PNG in S3 with cache headers.

**UC‑09: Rate Limiting / Abuse Protection**  
- Purpose: Prevent excessive trim jobs in a no‑auth environment.  
- Scope: IP‑based quotas, burst control, and friendly UI messaging.

**UC‑10: Accessibility & Fallbacks**  
- Purpose: Ensure the experience is operable by keyboard and resilient on older browsers.  
- Scope: WCAG‑aligned controls, clear focus states, and non‑canvas fallback UI.

---

### Glossary

- **Playhead** — The current playback position on the waveform.  
- **Selection box** — The draggable region that defines `startMs` and `endMs`.  
- **Presigned URL** — A time‑limited S3 link used to securely deliver the trimmed clip without auth.
