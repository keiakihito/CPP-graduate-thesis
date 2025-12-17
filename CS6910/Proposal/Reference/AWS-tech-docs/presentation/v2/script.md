## iPalpiti Audio Segmentation — Full Talking Script (20–30 minutes)

> Use this as spoken notes. Pacing guidance per section; adjust live. Keep tone candid and engineering-focused, with a nod to partial credit while emphasizing real stakeholder value.

---

### 0. Disclaimer and Expectations (2–3 min)
Hi professor, quick disclaimer before we dive in. This is a real stakeholder project for iPalpiti, an international orchestra association, not a classroom mock project. The scope goes out of the course rubric. If you grade strictly by the rubric, I understand I will get 0 because I don't use Java. Instead I used typescript. Also, my teammates aren’t in this class. I gave up other startup projects in this class to focus on this, so I am happy to accept any course grade including F. Partial credit is appreciated, though.

---

### 1. Why This Project / Mission (2–3 min)
The problem we’re solving: iPalpiti which is an internatioal music orchestra association needs a digital archive so fans can browse performances, preview, and download specific segments with payment. The mission is a web app where users pick a time range on a track, pay, and get a trimmed clip. The value is our project financially helps for this organization and each musicans credit, which is very important aspect in current era.

---

### 2. Team and Roles (2 min)
We split responsibilities by module instead of heavy Agile ceremonies to reduce learning curve for new tools:
- Frontend: Next.js UI for browsing, previewing, and selecting segments.
- Backend: Lambda HTTP API plus domain services—that’s my area.
- Database/Storage: PostgreSQL on RDS for metadata; S3 for raw audio assets.
Clear ownership let us learn AWS components faster and avoid stepping on each other.

---

### 3. High-Level AWS Architecture (3–4 min)
User flow: UI in Next.js hits API Gateway, which triggers Lambda. Lambda talks to RDS (PostgreSQL) for metadata and S3 for audio. S3 holds the raw mp4/wav files and the trimmed outputs; RDS stores album/track metadata plus S3 keys. Separating metadata from media keeps queries fast and storage simple. This stack keeps ops light: serverless API, managed DB, cheap object storage.

---

### 4. Backend Layered Intent (3–4 min)
Showing UML
We kept a simple three-layer design:
- Controllers: validate HTTP input/output and call services.
- Services: implement use cases like list albums, list tracks, preview, segment download.
- Infrastructure: repository interfaces with a Postgres implementation; swappable for other databases.
The goal is separation of concerns and dependency inversion. Controllers don’t know about Postgres; services depend on interfaces. If we ever swap AWS for local or Azure, we limit changes to infrastructure. It also makes testing easy—mock the repo, test services in isolation.

---

### 5. TDD Process and Habit (3–4 min)
My workflow is the classic loop: stubs → red bar → minimal green → refactor. In the spirit of Clean Code, tests are executable documentation and the safety net for refactoring; they keep functions small and responsibilities clear. The payoff: refactors stay safe and failures localize to a layer. 
In clean code, uncle bob address utlizing coverate tool to visualize test. Jest for unit tests plus coverage reports to visualize what’s truly exercised. So I will show how my current test result look like. The visualization Clean Code advocates—so we see exactly which paths we’re missing.

---

### 6. Coverage Snapshot (1–2 min)
Here is my test result in HTML format. Current coverage is about **78% statements / 66% branches**; branches remain the weak spot because I am still working on ffmpeg segmentation infra layer in AWS for debugging. 
My statements senction was bottle neck for this test coverage due to error handling  types. I created special object insetad of returning null, which causes a lot of if and else chain which I will show the later. It motivates me to apply the design pattern, Strategy refactor: isolate each error path instead of wrestling a big if/else chain, so branch coverage can climb.

---







### 6. Design Pattern #1 — Strategy (+ Singleton Registry) for Error Handling (4–5 min)
Problem: error handling was a growing if/else chain mapping domain errors to HTTP status codes. That breaks Open/Closed and makes branch coverage stall around 66%. Solution: an `ErrorMappingRegistry` that holds strategies—pure functions—per error class. For example, `AlbumNotFoundError` and `TrackNotFoundError` map to a not-found strategy returning 404; `InvalidSegmentSelectionError` maps to a bad-request strategy returning 400. We keep one shared registry instance so mappings are consistent; new errors are registered once. Benefits: higher branch coverage by testing each strategy directly, declarative mappings in one file, easy to extend without editing a giant conditional, and each strategy is unit-testable on its own.

---

### 7. Design Pattern #2 — Builder + Facade around FFmpeg (4–5 min)
Shoing codebase
We didn’t want FFmpeg flag soup leaking everywhere. The `FFmpegArgsBuilder` assembles command arguments from a `TrimRequest` and `AudioConfig`, so services and controllers never touch the flags. On top of that, `FFmpegAudioTrimmer` is a Facade exposing a simple `trim()` API—it hides command construction, process spawning, timeouts, and buffering. Supporting cast: `AudioSourceDownloader` adapts S3/local to “give me a local file and cleanup,” and `FFmpegProcessRunner` adapts process execution into a clean interface. Benefits: readability (one place understands FFmpeg), maintainability (change flags once), and testability (builder and facade are easy to unit-test).

---










### 11. Closing (1–2 min)
I appreciate watching until the last part of my presentaion. I learned so much in this project and Key takeaways: clean AWS separation, layered backend with DIP, TDD habit, Strategy for error mapping, Builder/Facade for FFmpeg, and a future-ready validation chain. Happy to dig into trade-offs, coverage targets, or AWS design choices. 

But especially TDD and Refactor process are crucial discovery in this project. As uncle bob says keep running all the tests and refactoring are primary habit to keep code clean and trustable. These idea never gets old. This is because no matter what AI builds app faster we are not sure the app is reliable. The tests show us the reliability. No matter how AI makes the codebase complexed, developper must understand codebase. Refactor helps for devellpers to understand codebase. If we igore theese factores the app will bring huge lost once it crushes. TDD and refactor helps a lot to understand the code base. That's what I got! Thanks for watching!
