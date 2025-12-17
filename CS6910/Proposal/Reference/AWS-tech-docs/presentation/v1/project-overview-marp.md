---
marp: true
theme: default
paginate: true
---

<!-- _class: lead -->
# iPalpiti Audio Segmentation
## Project Overview Presentation

**CS5800 F25**: Audio Segmentation 
Keita Katsumi
Elena Hernandez
Jay Wageni

Youtube:
https://youtu.be/-plWX0Rh-yM

---

**Note**: This project extends beyond the CS 5800 class scope but represents higher value work for this semester. This presentation demonstrates how course concepts are applied in a real-world project. Partial credit would be greatly appreciated.

---

## Slide 1: Project Mission & Stakeholder

### Stakeholder
**iPalpiti** - International Musician Association

### Team Mission
- **Create Digital Archive**: Preserve and organize musical performances
- **Audio Segmentation Download**: Enable users to download specific segments from audio tracks

### Project Goal
Build a web application that allows visitors to browse albums, preview tracks, select time ranges, and download trimmed audio clips.

---

## Slide 2: AWS Tech Stack Architecture

### System Flow
```
UI (Next.js) → Lambda (API Gateway) → RDS (PostgreSQL) ↔ S3 (Audio Storage)
```

### Components
- **S3**: Stores original audio files and temporary trimmed clips
- **RDS (PostgreSQL)**: Manages album and track metadata
- **Lambda**: Serverless API functions (HTTP API Gateway)
- **UI (Next.js)**: Frontend for browsing and downloading

### Data Flow
1. User requests album/track data → **Lambda** queries **RDS**
2. User selects audio segment → **Lambda** processes from **S3**
3. Trimmed clip generated → Stored in **S3**, URL returned to **UI**

---

## Slide 3: Complete Architecture Overview

### UML Architecture Diagram

![Architecture Overview](../diagram/UML/architecture-overview.png)

**Full System Architecture**:
- **API Layer**: Lambda HTTP Handler, Router, Controllers
- **Service Layer**: Business logic (Album, Track, Segmentation Services)
- **Domain Layer**: Domain models (Album, Track, SegmentSelection)
- **Infrastructure Layer**: Repository pattern (PostgresRepository)

---

## Slide 4: Architecture Layers (Simplified)

### Layered Design for Unit Testing

```
┌─────────────────────────────────────────┐
│         API Layer (Controllers)         │
│  - AlbumsController                     │
│  - TracksController                     │
│  - SegmentController                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Service Layer (Business Logic)     │
│  - AlbumService                         │
│  - TrackService                         │
│  - SegmentationService                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Infrastructure Layer (Repository)    │
│  - PostgresRepository                   │
└─────────────────────────────────────────┘
```

### Why This Architecture?
- **Separation of Concerns**: Each layer has a single responsibility
- **Unit Testing**: Each layer can be tested independently with mocks/stubs
- **Dependency Injection**: Services depend on repository interfaces, not implementations
- **Testability**: Mock infrastructure layer for fast, isolated unit tests

---

## Slide 5: Test-Driven Development (TDD) Approach

### TDD Cycle
```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐
│  Stubs  │ --> │ Red Bar  │ --> │ Green   │ --> │ Refactor │
│ (Write  │     │  (Write  │     │  Bar    │     │ (Improve │
│  Test)  │     │   Fails) │     │ (Make   │     │   Code)  │
└─────────┘     └──────────┘     └─────────┘     └──────────┘
     ▲                                                    │
     └────────────────────────────────────────────────────┘
```

### Process
1. **Stubs**: Create test structure with mocked dependencies
2. **Red Bar**: Write failing test (test first!)
3. **Green Bar**: Implement minimal code to pass test
4. **Refactor**: Improve code while keeping tests green

### Example Flow
```typescript
// 1. Stub: Create test with mock repository
const mockRepo = { getTrack: jest.fn() };
const service = new TrackService(mockRepo);

// 2. Red: Write failing test
it('returns track when found', async () => {
  mockRepo.getTrack.mockResolvedValue({ id: '1', title: 'Track' });
  const track = await service.getTrack('1');
  expect(track.title).toBe('Track'); // ❌ Fails - not implemented
});

// 3. Green: Implement minimal code
async getTrack(id: string) {
  return await this.repository.getTrack(id); // ✅ Passes
}
```

---

## Slide 6: Coverage Testing & Visualization

### Running Tests with Coverage
```bash
npm run test:coverage
```

### Current Coverage Metrics
- **Statements**: 91.39% (223/244) ✅
- **Branches**: 76.92% (80/104) ⚠️ **Lowest metric**
- **Functions**: 91.93% (57/62) ✅
- **Lines**: 92.95% (211/227) ✅

### Visualizing Coverage
```bash
# Generate HTML report
npm run test:coverage

# View interactive HTML report
npm run coverage:serve
# Opens http://localhost:8080
```

### HTML Report Features
- **File-by-file breakdown**: See coverage for each source file
- **Line-by-line highlighting**: 
  - 🟢 Green = Covered
  - 🔴 Red = Not covered
  - 🟡 Yellow = Partially covered (branches)
- **Branch coverage details**: See which if/else paths are tested

---

## Slide 7: Current Coverage Gap & Solution

### Problem: Uncovered Branches

**Current State**:
- **Branch Coverage**: 76.19% (lowest metric)

**Uncovered Branches**:
1. `logError` function: Context undefined path not tested
2. OR conditions: Not all branches separately tested
3. Fallback paths: Unmapped error handling difficult to test

**Root Cause**:
- No dedicated unit tests for error handling
- Error handling tested indirectly through controllers
- Hard to isolate and test individual branches

### Solution: Strategy Pattern

**Why Strategy Pattern?**
- ✅ **Improve Testability**: Each strategy is a pure function - easy to test
- ✅ **Higher Coverage**: Isolated components → easier to achieve 100% coverage
- ✅ **Reduce Duplication**: Reuse strategies for similar error types
- ✅ **Better Maintainability**: Clear separation of concerns

---

## Slide 8: Strategy Pattern Implementation

### Before (If-Chain)
```typescript
const mapErrorToResponse = (error: unknown) => {
  if (error instanceof AlbumNotFoundError || ...) {
    return { status: 404, message: error.message };
  }
  if (error instanceof TrackNotFoundError) {
    return { status: 404, message: error.message };
  }
  // ... more if statements
};
```

**Problems**:
- ❌ Hard to test each branch independently
- ❌ Must modify function for new errors
- ❌ Difficult to achieve high coverage

### After (Strategy Pattern)
```typescript
// Strategies (pure functions - easy to test)
const notFoundStrategy = (error) => ({ 
  status: 404, 
  message: error.message 
});

const badRequestStrategy = (error) => ({ 
  status: 400, 
  message: error.message 
});

// Registry (testable lookup logic)
class ErrorMappingRegistry {
  private strategies = new Map([
    [AlbumNotFoundError, notFoundStrategy],
    [TrackNotFoundError, notFoundStrategy],
    [InvalidSegmentSelectionError, badRequestStrategy],
  ]);
}
```

**Benefits**:
- ✅ Each strategy testable in isolation
- ✅ Registry lookup logic testable separately
- ✅ Easy to add new errors (just register)
- ✅ Higher test coverage achievable

---

## Slide 9: Expected Coverage Improvement

### Coverage Goals

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Branch Coverage** | 76.92% | **85-90%+** | +8-13% |
| **errorHandler.ts** | 91.66% | **98-100%** | +6-8% |
| **Overall Statements** | 91.39% | **93-95%+** | +2-4% |

### Testability Improvements

**Before**:
- ❌ No dedicated unit tests for error handling
- ❌ Indirect testing through controllers
- ❌ Hard to test edge cases

**After**:
- ✅ Dedicated test file for error handling
- ✅ Each strategy tested independently
- ✅ Registry tested separately
- ✅ Easy to test all branches systematically

---

## Slide 10: Summary

### Key Points

1. **Mission**: Digital archive + audio segmentation for iPalpiti
2. **Architecture**: AWS stack (S3 → RDS → Lambda → UI) with layered design
3. **Testing**: TDD approach with coverage visualization
4. **Challenge**: Branch coverage at 76.92% (lowest metric)
5. **Solution**: Strategy Pattern to improve testability and coverage

### Expected Outcomes

- **Better Testability**: Isolated, testable components
- **Higher Coverage**: 76.92% → 85-90%+ branch coverage
- **Reduced Duplication**: Reusable error mapping strategies
- **Maintainability**: Clear separation of concerns

### Next Steps

1. Implement Strategy Pattern in `errorHandler.ts`
2. Create dedicated unit tests for error handling
3. Achieve 98-100% coverage for error handling
4. Improve overall branch coverage to 85-90%+

---

<!-- _class: lead -->
# Thank You

**Implementaion**
- https://github.com/keiakihito/CS5800-P4-Submit




