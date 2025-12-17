# UC‑05 Plan (Inline ffmpeg Prototype)

## Scope and approach
- Keep everything inside `SegmentationService`: after validating `SegmentSelection`, trim audio via ffmpeg and return a `SegmentFile` (buffer or temp file).
- Use `PostgresRepository.getTrack()` to fetch `durationMs` and `s3Key`, download the source from S3, run `ffmpeg -ss … -to …`, and return a `SegmentFile`.
- API route (POST `/api/segment`) parses `{ trackId, startMs, endMs }`, calls `SegmentationService.generateClip`, and streams the file or returns a URL.
- Client handles download and surfaces validation errors.
- Operational watchouts: ffmpeg time/temp storage, enforce max selection, log track/range, keep S3 access minimal.
- Future: swap helper for Lambda/S3 or a job queue if inline trimming is slow, add more formats, or move to presigned URLs/CDN delivery.

## TDD incremental sprints

**Sprint 1: validation contract**
- [x] Add `SegmentSelection` validator that rejects non-numeric, negative, start>=end, and ranges exceeding track duration.
- [x] Tests: pure validation (red/green), edge cases for ms boundaries, error payload shape.

**Sprint 2: trimming helper seam**
- [x]Introduce `FFmpegTrimmer` interface (sync wrapper over child process) and inject into `SegmentationService`.
- [x]Tests: service uses seam; ffmpeg never invoked in unit tests; assert command args built correctly via mock.

**Sprint 3: repository + S3 integration**
- [x]Service fetches track (`durationMs`, `s3Key`) and downloads original to temp file/path before trimming.
- [x]Tests: stub repo + S3; reject missing track; reject selection beyond duration; ensure temp file cleanup hook called.

**Sprint 4: service happy path**
- [x]`generateClip` returns a `SegmentFile` descriptor (path or buffer + mime/size) from the mocked trimmer.
- [x]Tests: successful clip metadata; limits on max duration enforced; trims when start=0 and near-end.

**Sprint 5: API route**
- [x]Add POST `/api/segment` handler that maps request body -> `SegmentSelection`, calls service, streams result with headers.
- [x]Tests: request parsing errors -> 400; missing track -> 404; range errors -> 422; success -> 200 with stream headers.

**Sprint 6: client slice**
- UI flow to request segment and download/display errors.
- Tests: component/service mocks for API client; shows validation errors; triggers download on success.

## Test matrix to keep green
- Unit: validators, service with mocked trimmer/S3/repo, API route with supertest + mocks.
- Integration (opt-in): run ffmpeg against a fixture clip in CI if available; otherwise behind a feature flag.
- Ops checks: log duration and trim length; capture ffmpeg stderr for observability; guardrails on temp dir size.
