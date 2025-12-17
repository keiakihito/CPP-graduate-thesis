## Design Pattern Plan for iPalpiti Audio Segmentation Backend

**Document purpose**: Summarize how we apply GoF design patterns in the backend, with a clear priority on **simple, readable code** rather than “patterns for patterns’ sake.”  
This document is designed for the final presentation / report.

---

## 1. Prioritized Pattern: Strategy for Error Handling

### 1.1 Motivation

- **Problem**: Mapping domain errors to HTTP responses was previously a growing `if`/`else` chain in `errorHandler.ts`.
- **Issues**:
  - Violated **Open/Closed Principle** (every new error required modifying the same function).
  - Harder to test all branches and improve coverage.
  - Error-handling rules were not clearly visible in one place.
- **Goal**: Make error mapping **declarative, extensible, and easy to test** without adding too much complexity.

### 1.2 Strategy Pattern Design

- **Pattern**: GoF **Strategy** (Behavioral)
- **Intent**: “Define a family of algorithms, encapsulate each one, and make them interchangeable.”

- **In our context**:
  - **Strategy** = function that maps a specific error type to an HTTP response.
  - **Context** = `ErrorMappingRegistry` that holds all strategies and picks the right one at runtime.

**Key elements (conceptual):**

- `ErrorMappingStrategy`:  
  - Type/signature: `(error: Error) => { status: number; message: string }`
- Concrete strategies:
  - `notFoundStrategy` → 404 errors (e.g. `AlbumNotFoundError`, `TrackNotFoundError`).
  - `badRequestStrategy` → 400 errors (e.g. `InvalidSegmentSelectionError`).
- `ErrorMappingRegistry`:
  - Holds a map of `ErrorClass → ErrorMappingStrategy`.
  - Provides `map(error)` → returns corresponding HTTP error response or `null`.
  - Supports `register(errorClass, strategy)` for extension.

Controllers call:

- `handleControllerError(error, res, defaultMessage, context)`
  - Delegates to `mapErrorToResponse` → uses `errorRegistry`.
  - Sends mapped status/message or falls back to 500.

### 1.3 Benefits (Why this is priority #1)

- **Open/Closed Principle**:
  - New error type ⇒ **register** a strategy instead of modifying a long `if` chain.
- **Readability**:
  - All mappings live in one place (`ErrorMappingRegistry` setup).
  - It’s easy to scan: “this error → this status → this message.”
- **Testability & Coverage**:
  - Each strategy is a tiny, pure function → simple unit tests.
  - The registry can be tested independently:
    - Known error types → mapped correctly.
    - Unknown / non-`Error` values → return `null`.
  - `handleControllerError` + `logError` can be tested in isolation.
- **Low ceremony**:
  - The pattern is implemented with simple functions and one small class.
  - Controllers remain very simple; they just call `handleControllerError`.

**Summary**: Strategy for error mapping is a **high‑value, low‑complexity** improvement and is our **primary, “must-have” pattern**.

---

## 2. Patterns Already in Use (With Minimal Extensions)

This project already uses several GoF-style ideas naturally, especially around segmentation and FFmpeg integration.

### 2.1 Builder: FFmpeg Argument Construction

- **Where**: `FFmpegArgsBuilder`
- **Pattern**: GoF **Builder** (Creational)
- **Role**:
  - Encapsulates the logic to build FFmpeg command arguments from:
    - A **trim request** (`TrimRequest`) and
    - An **audio configuration** (`AudioConfig`).
  - Keeps the complex `string[]` command assembly out of services and controllers.

**Benefits:**

- **Readability**: Controllers/services never need to know FFmpeg flags (`-ss`, `-t`, `-ar`, etc.).
- **Maintainability**: When FFmpeg flags or formats change, only `FFmpegArgsBuilder` needs updating.
- **Testability**: We can unit-test one small class to verify FFmpeg command correctness.

**Planned extension (very small):**

- If we add more FFmpeg operations (e.g. concat, transcode):
  - Extend `FFmpegArgsBuilder` with additional methods, for example:
    - `buildConcatArgs(...)`
    - `buildTranscodeArgs(...)`
  - Still keep all FFmpeg-specific argument details in this single place.

### 2.2 Facade + Adapter: FFmpeg and Storage Integration

We already have a thin but effective separation of concerns around FFmpeg and storage.

#### 2.2.1 AudioTrimmer as a Small Facade

- **Where**:
  - Interface: `AudioTrimmer` (`trim(request: TrimRequest): Promise<SegmentFile>`)
  - Implementation: `FFmpegAudioTrimmer`
- **Pattern**: **Facade** (Structural)
- **Role**:
  - Provides a **simple, domain-friendly method** (`trim`) that hides:
    - FFmpeg command construction.
    - Process running / timeouts.
    - Output buffering and file naming.
  - `SegmentationService` and controllers only depend on `AudioTrimmer`, not on FFmpeg directly.

**Benefits:**

- Clear **separation** between domain logic and FFmpeg internals.
- Easier **debugging**: one focused place to look for FFmpeg issues.
- Potential to add features (logging, metrics) inside the facade without touching services.

**Lightweight extension (optional, still simple):**

- We can treat `FFmpegAudioTrimmer` explicitly as our **FFmpeg Facade**:
  - Keep a focused public surface (e.g. `trim` only, or `trim` + `probe` if needed).
  - Ensure no other parts of the codebase spawn FFmpeg directly.

#### 2.2.2 AudioSourceDownloader & Process Runner as Adapters

- **Where**:
  - `AudioSourceDownloader` with concrete `S3AudioSourceDownloader`.
  - `FFmpegProcessRunner` wrapping `child_process` (or similar).
- **Pattern**: **Adapter** (Structural)
- **Role**:
  - `AudioSourceDownloader` adapts different storage mechanisms to a simple interface:
    - “Give me a local path + a cleanup function.”
  - `FFmpegProcessRunner` adapts low-level process execution APIs to:
    - “Run this command with args and timeout; give me a `Buffer` or throw an error.”

**Benefits:**

- We already **decouple**:
  - Storage details (S3 vs local vs future storage) from the segmentation service.
  - Process execution details from FFmpeg-specific logic.
- This sets us up for future substitutions (new storage, remote FFmpeg, etc.) with minimal changes.

**Possible small extension (still simple):**

- Introduce additional `AudioSourceDownloader` implementations if needed:
  - `LocalFileAudioSourceDownloader` (using the local filesystem).
  - `HttpAudioSourceDownloader` (if we later stream from HTTP).
- Extend `FFmpegProcessRunner` if we want:
  - Better error parsing or structured logs.

### 2.3 Template-Method-Like Flow in SegmentationService

- **Where**: `SegmentationService.generateClip`
- **Pattern**: Conceptually similar to **Template Method** (Behavioral), but implemented as a straightforward method.
- **Flow**:
  1. `getTrackOrThrow`
  2. `validateSelectionOrThrow`
  3. `downloadAndTrim`

We currently keep this as a **simple, readable function** rather than a full abstract base class / subclass hierarchy. That matches our priority: clarity over pattern formality.

---

## 3. Optional Future Extensions for Scalability

These patterns are **not required now**, but the current design leaves room to add them later *only if the app’s requirements grow* (e.g., async jobs, multiple segmentation flows, richer monitoring).

### 3.1 Command Pattern for Segmentation Jobs

- **Motivation**:
  - If we later introduce:
    - **Asynchronous processing** (background jobs, queues),
    - **Retries** and **scheduling**,
    - Detailed **job audit logs** (who requested what, when, with which parameters).
- **Pattern**: **Command** (Behavioral)

**Conceptual design:**

- `SegmentAudioCommand`:
  - Encapsulates: track ID, time range, output format, destination, etc.
  - Exposes: `execute(): Promise<SegmentFile | void>`.
- `JobRunner` / `QueueWorker`:
  - Receives `SegmentAudioCommand` instances from a queue.
  - Coordinates execution and error handling.

**How it fits current code:**

- Internally, `execute()` would use the existing:
  - `SegmentationService`
  - `AudioTrimmer`
  - `AudioSourceDownloader`
- No need to redesign services; we simply wrap calls into Command objects for queueing.

### 3.2 Template Method for Multiple Segmentation Flows

- **Motivation**:
  - If we later support **different types of flows**, for example:
    - Standard “downloadable clip.”
    - Short **preview** clips (e.g., fixed max duration, different output format).
    - **Multi-track** or **crossfade** segments.
- **Pattern**: **Template Method** (Behavioral)

**Possible design:**

- `AbstractSegmentationFlow`:
  - Defines fixed steps:
    - `getTrack`
    - `validateSelection`
    - `downloadSource`
    - `runTrimmer`
    - `postProcessAndReturn`
  - Default implementations for common parts.
- Concrete flows:
  - `DownloadClipFlow`
  - `PreviewClipFlow`
  - `MixedSegmentFlow`

**Current decision:**

- For now, **we deliberately keep a single, clear `SegmentationService.generateClip`** method.
- Only when we truly have **multiple distinct flows** will we consider extracting a Template Method class hierarchy.

### 3.3 Chain of Responsibility for Validation Rules

- **Motivation**:
  - New validation rules may appear:
    - Selection within track duration.
    - Max length guardrail (`maxSelectionMs`).
    - User-based quotas or permissions.
    - Policy / copyright rules.
    - Future **payment / entitlement** checks (only paid/authorized users can download clips).
- **Pattern**: **Chain of Responsibility** (Behavioral)

**Conceptual design:**

- `SegmentValidationRule` interface:
  - `validate(track, selection): void` (throws a domain error on failure).
- Chain examples:
  - `DurationWithinTrackRule`
  - `MaxSelectionDurationRule`
  - `UserQuotaRule`
  - `PolicyRule`
- `SegmentationService`:
  - Holds a list of rules.
  - Calls each in sequence before performing segmentation.

#### 3.3.1 Concrete Example: Login + Payment + Segmentation Validation

A realistic future scenario for this project is:

1. Users **log in**.
2. Users **register a payment method** or have an approved payment/entitlement status (e.g., subscription, credits, or pay-per-download).
3. Only **authorized and entitled** users can download audio clips.

We can extend the validation chain to cover **both access control and technical segmentation rules** without exploding controller complexity.

**Example rule chain for a download request:**

1. `AuthenticatedUserRule`
   - Checks that the request has a valid authenticated user context (e.g., from JWT or session).
   - Throws a domain error like `UserNotAuthenticatedError` if missing/invalid.
2. `PaymentMethodRegisteredRule`
   - Verifies the user has a valid payment method or an approved payment profile.
   - Throws `PaymentMethodMissingError` if not set up.
3. `EntitledToDownloadRule`
   - Checks business rules: subscription active, sufficient credits, or paid access for the album/track.
   - Throws `UserNotEntitledToDownloadError` or `DownloadQuotaExceededError`.
4. `DurationWithinTrackRule`
   - Ensures the start/end times are inside the track duration.
5. `MaxSelectionDurationRule`
   - Enforces global guardrails (e.g., max 4-minute clip).

Each of these rules is a small, focused object/function implementing a common interface, for example:

```typescript
interface DownloadValidationRule {
  validate(context: {
    user: AuthenticatedUser | null;
    track: Track;
    selection: SegmentSelection;
  }): Promise<void> | void;
}
```

The **DownloadValidationChain** would then be responsible for invoking each rule in order:

```typescript
class DownloadValidationChain {
  constructor(private readonly rules: DownloadValidationRule[]) {}

  async validate(context: { user: AuthenticatedUser | null; track: Track; selection: SegmentSelection }) {
    for (const rule of this.rules) {
      await rule.validate(context); // May throw a domain error
    }
  }
}
```

`SegmentationService` (or an application-level service that orchestrates “download segment” use cases) would use this chain like:

```typescript
await downloadValidationChain.validate({ user, track, selection });
// If we reach here, user is authenticated, entitled, and the segment is valid.
```

**How this plays with our Strategy-based error handling:**

- Each rule throws **domain errors** (`UserNotAuthenticatedError`, `PaymentMethodMissingError`, etc.).
- The existing **Strategy registry** for error mapping can be extended to map these to:
  - `401 Unauthorized` for auth failures.
  - `402 Payment Required` / `403 Forbidden` / `429 Too Many Requests` for payment or quota issues.
- Controllers remain simple:
  - They just call the service method and rely on `handleControllerError` to convert any domain errors into HTTP responses.

**Why this fits our simplicity-first philosophy:**

- We start with only **a few clear rules** (auth, payment, entitlement, duration).
- Each rule is a small, readable unit that can be tested independently.
- The controller and `SegmentationService` code stays easy to follow.
- As more rules are added, we don’t add more `if`/`else` ladders; we just add new rule objects to the chain.

**Benefit:**

- Adding a new validation is just “append another rule object.”
- Error mapping Strategy will take care of surfacing them cleanly to HTTP responses.

### 3.4 Observer (Events) for Job Lifecycle (Optional)

- **Motivation**:
  - If we later need:
    - Real-time progress feedback (UI).
    - Structured logging / metrics (e.g., to CloudWatch, Datadog).
    - Loose coupling between core segmentation logic and monitoring.
- **Pattern**: **Observer** (Behavioral)

**Conceptual design:**

- `SegmentationEvents`:
  - Emits events like:
    - `jobStarted`
    - `jobProgress`
    - `jobCompleted`
    - `jobFailed`
- Observers:
  - Logging observer.
  - Metrics observer.
  - Optional UI / notification observer.

**Current decision:**

- For now, **simple logging inside `SegmentationService` / FFmpeg facade is enough**.
- We can refactor logs into event emissions once we have clear needs for multiple independent listeners.

---

## 4. Summary for Presentation

- **Primary pattern (implemented with highest priority)**:
  - **Strategy for error handling**:
    - Cleans up error mapping.
    - Boosts testability and coverage.
    - Follows Open/Closed Principle while staying simple.

- **Patterns already in the codebase**:
  - **Builder** for FFmpeg arguments (`FFmpegArgsBuilder`).
  - **Small Facade** for FFmpeg operations (`FFmpegAudioTrimmer` as the public face).
  - **Adapter** roles for storage and process execution (`AudioSourceDownloader`, `FFmpegProcessRunner`).
  - A clear, **Template-Method-like** segmentation flow in `SegmentationService.generateClip` (kept as a simple function for now).

- **Future extension points (optional)**:
  - **Command** pattern if we introduce async, queued segmentation jobs.
  - **Template Method** or Strategy variations if we get multiple distinct segmentation flows.
  - **Chain of Responsibility** for complex, composable validation rules.
  - **Observer** for job lifecycle events once we need richer logging/metrics/UX.

Overall, the design uses GoF patterns **only where they reduce complexity and improve readability**, and deliberately avoids unnecessary abstraction. This keeps the codebase friendly for future contributors while leaving clear growth paths as requirements evolve.


