# UC-01 – UC-04 TDD Implementation Plan

## Guiding Principles
- Work bottom-up: domain helpers → repository contract → services → API routes.
- Keep unit tests isolated with fakes/mocks; reserve real DB hit for integration tests.
- Maintain red-green-refactor cadence per step.

## Step-by-Step Breakdown

### 1. Domain Primitives ✅ 
1. Write failing tests for value helpers:
   - `Pagination` (page bounds, default page size logic if any).
   - `SegmentSelection` (validate start < end, duration limits for UC-04).
2. Implement minimal domain models (`Album`, `Track`) as TypeScript interfaces or classes.
3. Refactor as needed after tests pass.

### 2. Repository Contract✅
1. Define `PostgresRepository` interface with:
   - `listAlbums(page: number): Promise<AlbumSummary[]>`.
   - `getAlbum(id: string): Promise<AlbumDetail | null>`.
   - `listTracks(albumId: string): Promise<TrackSummary[]>`.
   - `getTrack(id: string): Promise<TrackDetail | null>` (needed for UC-03/04).
2. Unit test a simple in-memory/fake implementation to lock DTO shapes.
3. Leave real SQL implementation for later.

### 3. Service Layer (one use case at a time)
1. **UC-01 Browse Albums**✅
   - Write failing spec for `AlbumService.listAlbums` (covers pagination, empty list).
   - Implement using mocked repository; ensure errors bubble clearly.
2. **UC-02 View Tracks**✅
   - Add tests for `AlbumService.getAlbum` (404 handling) and `TrackService.listTracks` (empty vs populated).
3. **UC-03 Preview Track**✅
   - Test `TrackService.getTrack` returning metadata & waveform info (mock attachments).
4. **UC-04 Define Segment**✅
   - Expand `SegmentationService` tests around `validateSelection` using `SegmentSelection` helper and repository for `durationMs` lookup.
5. After each service test suite passes, refactor shared logic while keeping coverage.

### 4. API Routes (Next.js handlers)
1. For each service method, add route-level tests using supertest or Next.js testing utilities.
   - Mock injected services to verify HTTP status codes, JSON payloads, and error cases.
2. Implement route handlers after tests fail, then rerun to green.

### 5. Repository Implementation
1. Set up Postgres test database (Docker or local instance) with migration seeds.
2. Write integration tests executing repository methods against test data.
3. Implement SQL queries (via Prisma/Knex/Drizzle/pg) until tests pass; refactor for performance.

### 6. End-to-End Smoke Tests
1. Configure integration suite that wires: route → service → fake repository (in-memory) to confirm wiring per UC.
2. Optionally, add one full-stack test hitting the real repository with seeded data for sanity.

## Deliverables Checkpoints
- Domain helpers spec suite (green).
- Service suites for UC-01..UC-04 (green).
- API route tests (green) with mocked services.
- Repository integration suite (green).
- Lightweight E2E smoke tests executed in CI.

## Next Steps After UC-04
- Implement UC-05 per `doc/uc05-audio-segmentation-plan.md` using the established pattern.
- Introduce contract tests between services and the future `AudioClipGateway` as clipping backend matures.

