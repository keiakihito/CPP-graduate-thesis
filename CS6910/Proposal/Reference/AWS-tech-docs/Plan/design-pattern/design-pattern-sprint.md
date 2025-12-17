## Design Pattern Implementation Sprints (TDD-Oriented)

**Goal**: Introduce and extend design patterns in small, test-driven increments, keeping the codebase simple, readable, and well-covered by tests.  
We use a TDD loop for each step: **Stub → Red Bar → Green Bar → Refactor**.

---

## Sprint 1: Strategy Pattern for Error Handling (High Priority)

### 1.1 Objectives

- Replace the `if`/`else` error mapping logic with a Strategy-based `ErrorMappingRegistry`.
- Achieve high branch coverage for `errorHandler.ts`.
- Keep controllers unchanged except for using the updated error handler.

### 1.2 Tasks & TDD Steps

#### Task 1: Introduce core types and strategies

- **Stub**
  - [x] Create/adjust `ErrorResponse` type.
  - [x] Declare `ErrorMappingStrategy` type and `ErrorMappingRegistry` class (empty `map` implementation).
  - [x] Add basic strategies: `notFoundStrategy`, `badRequestStrategy` (initially unused or partially wired).
- **Red Bar**
  - [x] Write unit tests for:
    - `notFoundStrategy` returns `{ status: 404, message: error.message }`.
    - `badRequestStrategy` returns `{ status: 400, message: error.message }`.
  - [x] Tests fail because `map` or strategies are not fully implemented / not exported.
- **Green Bar**
  - [x] Implement and export strategies.
  - [x] Implement `ErrorMappingRegistry` with an internal `Map<Function, ErrorMappingStrategy>`.
  - [x] Make tests pass.
- **Refactor**
  - [x]Clean up naming, add clear comments that this is the Strategy pattern context.
  - [x] Ensure there is a single place where initial mappings are registered.

#### Task 2: Wire registry into `mapErrorToResponse`

- **Stub**
  - Create a `mapErrorToResponse` function that simply returns `null` or a placeholder.
  - Update `handleControllerError` to call `mapErrorToResponse` (even if it always returns `null` initially).
- **Red Bar**
  - Add unit tests for `mapErrorToResponse`:
    - Known errors (`AlbumNotFoundError`, `TrackNotFoundError`, `InvalidSegmentSelectionError`) must be mapped correctly.
    - Unknown errors and non-`Error` values return `null`.
  - Tests fail due to missing behavior.
- **Green Bar**
  - Connect `mapErrorToResponse` to the shared `errorRegistry` instance.
  - Register current domain errors with appropriate strategies.
  - Make all tests pass.
- **Refactor**
  - Confirm controllers (`albumsController`, `tracksController`, `segmentController`) still just call `handleControllerError`.
  - Remove old `if`/`else` mapping logic.
  - Improve test names and factor out common setup.

#### Task 3: Improve log and fallback coverage

- **Stub**
  - Ensure `logError` and `handleControllerError` are exported for testing (or test via a thin wrapper).
  - Add missing test skeletons for:
    - `logError` with and without `context`.
    - `handleControllerError` with mapped and unmapped errors.
- **Red Bar**
  - Write tests that explicitly check:
    - Console output format for `logError`.
    - 500 fallback for unmapped errors with `defaultMessage`.
  - Tests fail initially (e.g., no logging or wrong message).
- **Green Bar**
  - Adjust `logError` to use the `context ? "<context> error" : "Controller error"` pattern.
  - Ensure `handleControllerError` uses mapping first, fallback to logging + 500.
  - Make tests pass.
- **Refactor**
  - Use `jest.spyOn(console, "error")` (or equivalent) with proper restore.
  - Confirm branch coverage meets or exceeds target (e.g. ~98–100% for `errorHandler.ts`).

---

## Sprint 2: Consolidate FFmpeg Builder and Facade (Small, Safe Improvements)

### 2.1 Objectives

- Keep FFmpeg details isolated behind `FFmpegArgsBuilder` and `FFmpegAudioTrimmer`.
- Make it easy to evolve FFmpeg behavior without touching controllers/services.
- Avoid over-engineering (no unnecessary new abstractions).

### 2.2 Tasks & TDD Steps

#### Task 1: Strengthen `FFmpegArgsBuilder` tests

- **Stub**
  - Identify all current argument patterns in `buildTrimArgs`.
  - Add test skeletons covering:
    - Correct use of `-ss`, `-t`, input path, and audio parameters.
    - Mapping from `outputExtension` to FFmpeg format (wav, mp3, m4a, aac).
- **Red Bar**
  - Implement tests that assert the resulting `string[]` includes:
    - Correct start time and duration in seconds.
    - Correct codec, sample rate, and channels based on `AudioConfig`.
    - Correct `-f` format for each supported extension.
  - Tests fail initially if coverage is incomplete or assumptions don’t hold.
- **Green Bar**
  - Adjust `FFmpegArgsBuilder` if necessary (minimal changes).
  - Make tests pass, ensuring behavior is clearly locked in.
- **Refactor**
  - Simplify test helpers for building `TrimRequest` and `AudioConfig`.
  - Keep `FFmpegArgsBuilder` small and focused.

#### Task 2: Treat `FFmpegAudioTrimmer` as the main FFmpeg Facade

- **Stub**
  - Add aFocused tests for `FFmpegAudioTrimmer.trim`:
    - That it delegates correctly to `FFmpegArgsBuilder` and `FFmpegProcessRunner`.
    - That it returns a `SegmentFile` with expected filename and content type.
  - Use a fake or mock `FFmpegProcessRunner` to avoid real FFmpeg calls in unit tests.
- **Red Bar**
  - Tests fail because the current implementation might not be fully observable or mocks not wired yet.
- **Green Bar**
  - Inject or mock `FFmpegProcessRunner` and verify:
    - It is called once with the expected options (command, args, timeout).
    - Output buffer is passed through to `SegmentFile.data`.
  - Make tests pass.
- **Refactor**
  - Add comments clarifying that `FFmpegAudioTrimmer` is the “FFmpeg Facade” for trimming.
  - Ensure no other module spawns FFmpeg directly (centralize FFmpeg usage here).

---

## Sprint 3: Validation and Future Chain of Responsibility (Including Payment Scenario)

### 3.1 Objectives

- Prepare for richer validation rules without complicating controllers.
- Lay out a path to Chain of Responsibility for:
  - Authentication.
  - Payment / entitlement logic.
  - Technical segmentation constraints.

### 3.2 Short-Term: Keep Validation Simple but Testable

#### Task 1: Formalize current validation rules (without full CoR yet)

- **Stub**
  - Clearly isolate the existing selection validation in `SegmentationService`:
    - `validateSelectionOrThrow` and `isSelectionValid`.
  - Add test skeletons for:
    - Valid selection (within track duration and below `maxSelectionMs`).
    - Invalid selection (outside duration or over max).
- **Red Bar**
  - Implement tests that capture current behavior (off-by-one boundaries, edge cases).
  - Any gaps cause failing tests.
- **Green Bar**
  - Adjust implementation only if tests reveal missing conditions or bugs.
  - Make tests pass.
- **Refactor**
  - Improve naming and comments around validation.
  - Keep everything in a single, readable method for now.

### 3.3 Longer-Term: Introduce Chain of Responsibility When Needed

This phase is **optional** and only triggered if/when we implement login + payment + entitlement features.

#### Task 2: Define validation rule interface & basic chain

- **Stub**
  - Introduce a `DownloadValidationRule` interface:
    - `validate(context: { user: AuthenticatedUser | null; track: Track; selection: SegmentSelection }): Promise<void> | void;`
  - Create an initial `DownloadValidationChain` class with a no-op `validate` method.
- **Red Bar**
  - Write tests:
    - That the chain calls each rule in order.
    - That if a rule throws an error, subsequent rules are not called.
  - Tests fail because `validate` is not implemented.
- **Green Bar**
  - Implement `DownloadValidationChain.validate` to iterate over rules and await each one.
  - Make tests pass.
- **Refactor**
  - Keep the chain small and straightforward.
  - Add comments explaining that this is a Chain of Responsibility for future extensibility.

#### Task 3: Add concrete rules for auth + payment + entitlement

- **Stub**
  - Define domain errors (names only, or full classes when needed), e.g.:
    - `UserNotAuthenticatedError`
    - `PaymentMethodMissingError`
    - `UserNotEntitledToDownloadError`
    - `DownloadQuotaExceededError`
  - Create rule classes (initially with empty `validate` bodies):
    - `AuthenticatedUserRule`
    - `PaymentMethodRegisteredRule`
    - `EntitledToDownloadRule`
    - `DownloadQuotaRule`
- **Red Bar**
  - For each rule, write tests that:
    - It throws the appropriate domain error when the condition is not met.
    - It passes silently when the condition is satisfied.
  - Tests fail because `validate` is not implemented.
- **Green Bar**
  - Implement each `validate` method using the context (user, subscription state, quotas).
  - Make tests pass.
- **Refactor**
  - Ensure error messages are clear and consistent.
  - Keep each rule small and focused (single responsibility).

#### Task 4: Integrate validation chain into the segmentation/download flow

- **Stub**
  - Adjust the application service that handles “download segment” to:
    - Accept a `DownloadValidationChain` dependency.
    - Call `downloadValidationChain.validate(context)` early in the flow.
  - Add integration-style tests that simulate:
    - Unauthenticated user.
    - No payment method.
    - Not entitled user.
    - Valid, entitled user.
- **Red Bar**
  - Tests fail initially because the chain is not wired or not invoked.
- **Green Bar**
  - Wire the chain into the service.
  - Make integration tests pass by observing thrown domain errors.
- **Refactor**
  - Keep controllers thin: they just call the service and rely on existing `handleControllerError`.
  - Move any duplicated validation logic from controllers into the chain.

#### Task 5: Extend error Strategy registry for new domain errors

- **Stub**
  - Add entries in the Strategy-based error registry plan for new errors:
    - Map `UserNotAuthenticatedError` to 401.
    - Map payment/entitlement errors to 402/403/429 as appropriate.
  - Write tests for `mapErrorToResponse` / registry:
    - That each new error maps to the expected status and message.
- **Red Bar**
  - Tests fail because errors are not registered.
- **Green Bar**
  - Register the new domain errors with appropriate strategies (e.g., `unauthorizedStrategy`, `paymentRequiredStrategy`, etc., or reuse `badRequestStrategy` where it makes sense).
  - Make tests pass.
- **Refactor**
  - Group related registry registrations (auth, payment, entitlement) together for readability.
  - Keep the Strategy registry declarative and easy to scan.

---

## Sprint 4 (Optional): Prepare for Async Commands & Observers

This sprint is only needed if we move to true async / queued jobs.

### 4.1 Command Pattern for Background Jobs

- **Stub**
  - Define a `SegmentAudioCommand` interface/class with `execute()` method.
  - Add test skeletons for:
    - Executing the command calls `SegmentationService.generateClip` (or equivalent).
    - Errors are propagated as domain errors.
- **Red Bar**
  - Tests fail because `execute` is not wired.
- **Green Bar**
  - Implement `execute` using existing services.
  - Make tests pass.
- **Refactor**
  - Keep the command simple: it should just hold parameters and delegate to services.

### 4.2 Observer for Job Lifecycle Events (Optional)

- **Stub**
  - Define a simple event emitter or observer interface for job lifecycle events (`started`, `completed`, `failed`).
  - Add test skeletons that:
    - Observers are notified in the correct order.
    - Errors in observers do not break core logic.
- **Red Bar**
  - Tests fail because no notifications are sent.
- **Green Bar**
  - Emit events at key points in command execution or service flow.
  - Make tests pass.
- **Refactor**
  - Keep observers optional and loosely coupled (easy to enable/disable).

---

## Summary

- **Sprint 1**: Implement the Strategy pattern for error handling with strong TDD, improving coverage and clarity.
- **Sprint 2**: Consolidate and test the FFmpeg Builder/Facade layer, keeping FFmpeg complexity isolated.
- **Sprint 3**: Strengthen current validation and define a clear path to a Chain of Responsibility for auth, payment, and segmentation rules, integrated with the Strategy-based error mapping.
- **Sprint 4 (optional)**: Prepare for future async jobs (Command) and richer monitoring (Observer) if and when requirements grow.

Each sprint is intentionally small and test-driven, so the codebase remains understandable while gaining the benefits of selected GoF patterns.


