## iPalpiti Audio Segmentation – 20–30 Minute Presentation Script (v2)

> **Tone**: candid, engineering-focused, aiming for partial credit while showing real project value for iPalpiti.  
> **Audience**: class evaluators + anyone curious about the real-world delivery.  
> **Format**: talk-track with timing cues; adapt phrasing live.

---

### 0. Disclaimer and Expectations (2–3 min)
- This is a real stakeholder project for **iPalpiti** (international orchestra association), not a classroom mock project.  
- Scope extends past course requirements; if grading strictly by the rubric, I understand a low score. Partial credit is appreciated.  
- Team members aren’t in this class; responsibility for course alignment is mine alone. I paused other startup work to focus on this.

### 1. Why This Project / Mission (2–3 min)
- **Problem**: iPalpiti needs a **digital archive** so people can browse performances, preview, and download specific segments.  
- **Mission**: build a web app where users select a time range on a track and get a trimmed clip.  
- **Value**: preserves cultural assets and enables monetization via paid downloads.

### 2. Team and Roles (2 min)
- We split by module instead of strict Agile ceremonies to reduce ramp-up:  
  - **Frontend**: Next.js UI for browsing, previewing, selecting segments.  
  - **Backend**: Lambda HTTP API + domain services (my part).  
  - **Database/Storage**: PostgreSQL on RDS for metadata; S3 for raw audio assets.  
- This division enforced clear ownership and faster learning on AWS components.

### 3. High-Level AWS Architecture (3–4 min)
- User flow: **UI (Next.js)** → **API Gateway → Lambda** → **RDS (PostgreSQL)** ↔ **S3**.  
- S3 keeps raw audio (mp4/wav) and trimmed outputs; RDS stores album/track metadata and S3 keys.  
- Keeping metadata separate from raw media reduces storage complexity and keeps queries fast.

### 4. Backend Layered Intent (3–4 min)
- **Controller layer**: validates HTTP I/O, forwards to services.  
- **Service layer**: realizes use cases (list albums, tracks, preview, segment download).  
- **Infrastructure layer**: repository interfaces with a Postgres implementation; swappable for other databases (DIP).  
- Goals: separation of concerns, easy dependency injection for tests, minimal change blast radius when swapping infra (AWS → local/Azure).

### 5. TDD Process and Habit (3–4 min)
- Workflow: **stubs → red bar → minimal green → refactor**.  
- Benefits: executable documentation for teammates, fast feedback during refactors, clear fault isolation per layer.  
- Coverage as of last run (from v1): ~91% statements, ~77% branches; strategy work targets the weakest metric (branches).  
- Tools: Jest + coverage HTML reports; aim to keep tests green while refactoring.

### 6. Design Pattern #1 – Strategy (+ Singleton Registry) for Error Handling (4–5 min)
- **Problem**: a growing `if/else` chain mapping domain errors to HTTP responses violated Open/Closed and was hard to cover.  
- **Solution**: `ErrorMappingRegistry` holds **strategies** (pure functions) per error class (e.g., `AlbumNotFoundError → 404`, `InvalidSegmentSelectionError → 400`).  
- **Singleton flavor**: one shared registry instance keeps mappings consistent across controllers; new errors register once.  
- **Benefits**: higher branch coverage, isolated tests per strategy, declarative mappings in one file, easy extensibility without editing a giant conditional.

### 7. Design Pattern #2 – Builder + Facade around FFmpeg (4–5 min)
- **Builder**: `FFmpegArgsBuilder` assembles command args from a `TrimRequest` and `AudioConfig` so services/controllers never touch FFmpeg flags.  
- **Facade**: `FFmpegAudioTrimmer` exposes a simple `trim()` API hiding command construction, process spawning, timeouts, and buffering.  
- **Adapters in play**: `AudioSourceDownloader` (S3/local) and `FFmpegProcessRunner` wrap storage/process details.  
- **Benefits**: readability (one place knows FFmpeg flags), maintainability (change flags once), testability (unit-test builder/facade in isolation).

### 8. Future Pattern – Chain of Responsibility for Validation (3–4 min)
- **Why**: as we add auth, payment/entitlement, quotas, and stricter segment rules, a flat `if` chain will explode.  
- **Plan**: compose small `ValidationRule` objects (auth check → payment check → entitlement → duration guards). Each throws a domain error.  
- **Fit**: errors plug into the existing Strategy registry (`401/402/403/429`), keeping controllers slim and flows composable.  
- **Benefit**: add/remove rules without touching controller/service core logic; each rule is unit-testable.

### 9. Major Features Delivered / In Progress (2–3 min)
- Core segmentation flow: browse albums/tracks, select segment, download trimmed clip.  
- AWS wiring: API Gateway + Lambda + RDS + S3 with metadata/raw separation.  
- Testing habit: TDD loop, high coverage focus, strategy refactor to raise branch coverage.  
- Design patterns applied pragmatically (not “patterns for patterns’ sake”).

### 10. Risks, Lessons, and Rationale (2–3 min)
- Risk: real-stakeholder scope exceeds class rubric; time spent on AWS ramp-up.  
- Lesson: TDD + small, testable units (strategy/builder/facade) kept refactors safe.  
- Rationale: choose **simple patterns that reduce complexity**; defer heavier patterns (Command/Observer) until real needs emerge.

### 11. Closing (1–2 min)
- Thanks for considering partial credit given the real-world scope.  
- Key takeaways: separation of concerns on AWS, TDD discipline, Strategy for error mapping, Builder/Facade for FFmpeg, future-ready validation chain.  
- Open to questions on trade-offs, coverage targets, or AWS architecture choices.
