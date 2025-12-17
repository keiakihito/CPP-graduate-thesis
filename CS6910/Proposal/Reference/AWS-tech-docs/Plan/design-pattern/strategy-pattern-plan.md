# Strategy Pattern for Error Mapping

**Presentation Document for Design Pattern Implementation**

---

## 1. Introduction

### What is the Strategy Pattern?

The **Strategy Pattern** is one of the classic behavioral design patterns from the **Gang of Four (GoF)** book "Design Patterns: Elements of Reusable Object-Oriented Software" (1994).

**Definition**: The Strategy Pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**Key Concepts**:
- Define a family of behaviors (strategies)
- Encapsulate each behavior as an object
- Make behaviors interchangeable at runtime
- Decouple algorithm implementation from code that uses it

### GoF Classification

- **Type**: Behavioral Pattern
- **Purpose**: Manage algorithms/behaviors
- **Applicability**: When you need different variants of an algorithm and want to switch between them dynamically

---

## 2. Current Implementation (Before Strategy Pattern)

### Problem Statement

Our error handling in `errorHandler.ts` uses a **conditional chain** (if-else ladder) to map domain errors to HTTP responses:

```typescript
// Current: backend/src/api/controllers/errorHandler.ts
const mapErrorToResponse = (error: unknown): ErrorResponse | null => {
  if (error instanceof AlbumNotFoundError || error instanceof TracksNotFoundFromAlbumError) {
    return { status: 404, message: error.message };
  }
  if (error instanceof TrackNotFoundError) {
    return { status: 404, message: error.message };
  }
  if (error instanceof InvalidSegmentSelectionError) {
    return { status: 400, message: error.message };
  }
  return null;
};
```

### Issues with Current Approach

1. **Violates Open/Closed Principle**: Must modify function every time we add a new error type
2. **Growing Complexity**: If-chain grows linearly with error types (currently 4 types, will grow)
3. **Poor Maintainability**: Hard to see all error mappings at a glance
4. **Testing Difficulty**: Each new error requires updating the same function
5. **Code Duplication**: Similar logic repeated in if statements
6. **No Extensibility**: Cannot easily add custom mappings or error transformations
7. **Poor Test Coverage**: Hard to achieve high branch coverage due to tightly coupled conditional logic

### Current Error Types

```typescript
// backend/src/domain/errors.ts
export class AlbumNotFoundError extends Error { ... }        // → 404
export class TracksNotFoundFromAlbumError extends Error { ... } // → 404
export class TrackNotFoundError extends Error { ... }        // → 404
export class InvalidSegmentSelectionError extends Error { ... } // → 400
```

### Usage in Controllers

All three controllers use the same error handler:

```typescript
// albumsController.ts
catch (error) {
  return handleControllerError(error, res, "Failed to fetch albums", "albumsController.list");
}

// tracksController.ts
catch (error) {
  return handleControllerError(error, res, "Failed to fetch tracks", "tracksController.listByAlbum");
}

// segmentController.ts
catch (error) {
  return handleControllerError(error, res, "Failed to process segment", "segmentController.download");
}
```

### Current Test Coverage Issues

**Problem**: Our current code coverage analysis reveals testability challenges:

- **Overall Branch Coverage**: 76.92% (80/104 branches) - **Lowest metric**
- **errorHandler.ts Branch Coverage**: 91.66% - Missing critical branches
- **No Dedicated Unit Tests**: Error handling is only tested indirectly through controller tests

**Uncovered Branches Identified**:

1. **Line 28 - `logError` function**:
   ```typescript
   const logMessage = context ? `${context} error` : "Controller error";
   ```
   - ✅ Branch 1 (context provided): Tested
   - ❌ Branch 2 (context undefined): **NOT TESTED** - Hard to test in isolation

2. **Line 11 - OR condition**:
   ```typescript
   if (error instanceof AlbumNotFoundError || error instanceof TracksNotFoundFromAlbumError)
   ```
   - Both error types map to 404, but OR branches may not be separately tested

3. **Line 38-39 - Fallback path**:
   ```typescript
   if (mapped) {
     return sendErrorResponse(...);
   }
   // else path (unmapped errors) - difficult to test systematically
   ```

**Why Current Approach Fails for Testing**:

- **No Direct Unit Tests**: `errorHandler.ts` has no dedicated test file
- **Indirect Testing Only**: Error handling tested through controller integration tests
- **Hard to Isolate**: Can't test individual branches without complex test setup
- **Difficult Edge Cases**: Testing `context` undefined requires creating full controller mocks
- **Growing Test Complexity**: Each new error type makes the test suite more complex

**Coverage Goal**: Achieve **90%+ branch coverage** for error handling code

---

## 3. Proposed Implementation (With Strategy Pattern)

### Strategy Pattern Architecture

```
┌─────────────────────────────────────────────────────┐
│  ErrorMappingStrategy (Interface/Type)              │
│  - map(error: Error): ErrorResponse                 │
└─────────────────────────────────────────────────────┘
                        ▲
                        │ implements
        ┌───────────────┼───────────────┐
        │               │               │
┌───────────────┐ ┌─────────────┐ ┌──────────────────┐
│ NotFoundError │ │ BadRequest  │ │ Custom Strategy  │
│   Strategy    │ │  Strategy   │ │   (Future)       │
└───────────────┘ └─────────────┘ └──────────────────┘

                        ▼
        ┌───────────────────────────────────┐
        │  ErrorMappingRegistry (Context)   │
        │  - strategies: Map<Type, Strategy>│
        │  - map(error): ErrorResponse      │
        └───────────────────────────────────┘
```

### Implementation Code

```typescript
// backend/src/api/controllers/errorHandler.ts (UPDATED)
import {
  AlbumNotFoundError,
  TracksNotFoundFromAlbumError,
  TrackNotFoundError,
  InvalidSegmentSelectionError,
} from "../../domain/errors.js";

type ErrorResponse = { status: number; message: string };

// ============================================
// Strategy Pattern: Error Mapping Strategy
// ============================================

/**
 * Strategy Interface: Defines how to map an error to HTTP response
 */
type ErrorMappingStrategy = (error: Error) => ErrorResponse;

/**
 * Concrete Strategy: Maps 404 Not Found errors
 */
const notFoundStrategy: ErrorMappingStrategy = (error) => ({
  status: 404,
  message: error.message,
});

/**
 * Concrete Strategy: Maps 400 Bad Request errors
 */
const badRequestStrategy: ErrorMappingStrategy = (error) => ({
  status: 400,
  message: error.message,
});

/**
 * Context: Registry that holds all error mapping strategies
 * This is the Strategy Pattern's "Context" that manages strategies
 */
class ErrorMappingRegistry {
  private strategies = new Map<Function, ErrorMappingStrategy>([
    // Register strategies for each error type
    [AlbumNotFoundError, notFoundStrategy],
    [TracksNotFoundFromAlbumError, notFoundStrategy],
    [TrackNotFoundError, notFoundStrategy],
    [InvalidSegmentSelectionError, badRequestStrategy],
  ]);

  /**
   * Register a new error mapping strategy
   * Allows extension without modifying existing code (Open/Closed Principle)
   */
  register(errorClass: Function, strategy: ErrorMappingStrategy): void {
    this.strategies.set(errorClass, strategy);
  }

  /**
   * Map error to response using registered strategy
   */
  map(error: unknown): ErrorResponse | null {
    if (!(error instanceof Error)) return null;

    // Find and execute the appropriate strategy
    for (const [ErrorClass, strategy] of this.strategies) {
      if (error instanceof ErrorClass) {
        return strategy(error);
      }
    }

    return null;
  }
}

// ============================================
// Singleton instance of the registry
// ============================================
const errorRegistry = new ErrorMappingRegistry();

/**
 * Updated: Uses Strategy Pattern to map errors
 */
const mapErrorToResponse = (error: unknown): ErrorResponse | null => {
  return errorRegistry.map(error);
};

// Rest of the code remains the same...
const sendErrorResponse = (res: any, status: number, message: string): any => {
  return res.status(status).json({ message });
};

const logError = (error: unknown, context?: string): void => {
  const logMessage = context ? `${context} error` : "Controller error";
  console.error(logMessage, error);
};

export const handleControllerError = (
  error: unknown,
  res: any,
  defaultMessage: string,
  context?: string
): any => {
  const mapped = mapErrorToResponse(error);
  if (mapped) {
    return sendErrorResponse(res, mapped.status, mapped.message);
  }

  logError(error, context);
  return sendErrorResponse(res, 500, defaultMessage);
};

// Export for extensibility
export { errorRegistry };
```

---

## 4. Benefits of Strategy Pattern Implementation

### ✅ 1. Open/Closed Principle (SOLID)

**Before**: Must modify `mapErrorToResponse` function for every new error type

```typescript
// Adding new error requires modifying this function
const mapErrorToResponse = (error: unknown): ErrorResponse | null => {
  // ... existing if statements ...
  if (error instanceof NewErrorType) { // ← MODIFICATION
    return { status: 418, message: error.message };
  }
};
```

**After**: Add new errors by registering strategies (no modification)

```typescript
// Adding new error is just registration (extension)
errorRegistry.register(NewErrorType, (error) => ({
  status: 418,
  message: error.message,
}));
```

### ✅ 2. Improved Maintainability

**Before**: Linear growth of if-chain (O(n) complexity)
- 4 error types = 3 if statements
- 10 error types = 9 if statements
- Hard to scan and understand

**After**: Declarative mapping in one place
- All mappings visible in the Map constructor
- Easy to scan and understand at a glance
- Constant-time lookup O(1) with Map

### ✅ 3. Better Testability & Improved Coverage

**Primary Motivation**: The Strategy Pattern significantly improves testability, making it easier to achieve higher code coverage percentages.

#### Current Testability Problems

**Before**: Test one large function with many branches - difficult to achieve high coverage

```typescript
// Must test all branches in one function - hard to isolate
describe('mapErrorToResponse', () => {
  it('maps AlbumNotFoundError');
  it('maps TrackNotFoundError');
  it('maps InvalidSegmentSelectionError');
  it('handles unknown errors');
  // Hard to test: context undefined, OR conditions, edge cases
});
```

**Problems**:
- ❌ No dedicated unit tests for `errorHandler.ts`
- ❌ Error handling tested indirectly through controllers
- ❌ Hard to test individual branches in isolation
- ❌ Current branch coverage: **91.66%** (missing critical branches)
- ❌ Overall project branch coverage: **76.92%** (lowest metric)

#### Strategy Pattern Solution

**After**: Test strategies independently - easy to achieve 100% coverage

```typescript
// Test each strategy in complete isolation
describe('notFoundStrategy', () => {
  it('returns 404 status with error message', () => {
    const error = new AlbumNotFoundError('album-1');
    const result = notFoundStrategy(error);
    expect(result).toEqual({ status: 404, message: 'Album not found: album-1' });
  });
});

describe('badRequestStrategy', () => {
  it('returns 400 status with error message', () => {
    const error = new InvalidSegmentSelectionError('track-1');
    const result = badRequestStrategy(error);
    expect(result).toEqual({ status: 400, message: 'Invalid segment selection for track: track-1' });
  });
});

// Test registry lookup logic separately
describe('ErrorMappingRegistry', () => {
  it('finds correct strategy for AlbumNotFoundError', () => {
    const registry = new ErrorMappingRegistry();
    const error = new AlbumNotFoundError('1');
    const result = registry.map(error);
    expect(result?.status).toBe(404);
  });

  it('returns null for unknown error types', () => {
    const registry = new ErrorMappingRegistry();
    const error = new Error('Unknown');
    expect(registry.map(error)).toBeNull();
  });

  it('returns null for non-Error objects', () => {
    const registry = new ErrorMappingRegistry();
    expect(registry.map('string')).toBeNull();
    expect(registry.map(null)).toBeNull();
    expect(registry.map(undefined)).toBeNull();
  });
});

// Test logError branches easily
describe('logError', () => {
  it('uses default message when context is undefined', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    logError(new Error('test'), undefined);
    expect(consoleSpy).toHaveBeenCalledWith('Controller error', expect.any(Error));
    consoleSpy.mockRestore();
  });

  it('uses context when provided', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    logError(new Error('test'), 'myContext');
    expect(consoleSpy).toHaveBeenCalledWith('myContext error', expect.any(Error));
    consoleSpy.mockRestore();
  });
});
```

#### Coverage Improvement Benefits

| Aspect | Before (If-Chain) | After (Strategy Pattern) |
|--------|-------------------|--------------------------|
| **Branch Coverage** | 91.66% (missing branches) | **~98-100%** (estimated) |
| **Test Isolation** | Poor (indirect testing) | Excellent (isolated units) |
| **Test Maintainability** | Low (complex setup) | High (simple, focused tests) |
| **Edge Case Testing** | Difficult | Easy (pure functions) |
| **Adding Test Cases** | Risky (modify existing) | Safe (add new test files) |

#### Specific Coverage Improvements

**Uncovered Branches → Easy to Test**:

1. **Line 28 (context undefined)**:
   - **Before**: Requires full controller mock setup
   - **After**: Simple unit test with `logError(error, undefined)`

2. **OR Condition Branches**:
   - **Before**: Hard to test each OR branch separately
   - **After**: Test each error type independently with its strategy

3. **Unknown Error Handling**:
   - **Before**: Difficult to test fallback path
   - **After**: Test registry with unknown error types directly

4. **Non-Error Objects**:
   - **Before**: Rarely tested edge case
   - **After**: Easy to test with `registry.map('string')`, `registry.map(null)`

#### Expected Coverage Results

**Projected Improvements**:
- **errorHandler.ts Branch Coverage**: 91.66% → **98-100%**
- **Overall Project Branch Coverage**: 76.92% → **85-90%+**
- **Test File Count**: 0 dedicated tests → **1 comprehensive test file**
- **Test Maintainability**: Low → **High**

**Why Strategy Pattern Achieves Higher Coverage**:

1. **Pure Functions**: Each strategy is a pure function - easy to test all paths
2. **Isolated Components**: Registry, strategies, and handlers tested separately
3. **Clear Test Boundaries**: Each component has well-defined responsibilities
4. **Edge Case Testing**: Easy to test null, undefined, unknown types
5. **Systematic Coverage**: Can write tests to cover every branch methodically

### ✅ 4. Extensibility & Flexibility

**Custom strategies for special cases**:

```typescript
// Add custom logging strategy for security errors
errorRegistry.register(SecurityError, (error) => {
  securityLogger.alert(error); // Custom behavior
  return { status: 403, message: 'Access denied' };
});

// Add rate limiting strategy
errorRegistry.register(RateLimitError, (error) => {
  return { 
    status: 429, 
    message: error.message,
    retryAfter: error.retryAfter // Extra fields
  };
});
```

### ✅ 5. Reduced Code Duplication

**Before**: Repeated pattern for similar errors

```typescript
if (error instanceof AlbumNotFoundError || error instanceof TracksNotFoundFromAlbumError) {
  return { status: 404, message: error.message }; // Duplicated
}
if (error instanceof TrackNotFoundError) {
  return { status: 404, message: error.message }; // Duplicated
}
```

**After**: Reuse the same strategy

```typescript
const notFoundStrategy = (error) => ({ status: 404, message: error.message });
// Reused for multiple error types
[AlbumNotFoundError, notFoundStrategy],
[TracksNotFoundFromAlbumError, notFoundStrategy],
[TrackNotFoundError, notFoundStrategy],
```

### ✅ 6. Clear Separation of Concerns

- **Strategy**: How to map a single error type
- **Registry**: Which strategy to use for which error
- **Handler**: Orchestration and fallback logic

---

## 5. Future Extensibility Examples

### Example 1: Add Validation Errors

```typescript
// New error type
export class ValidationError extends Error {
  constructor(public fields: string[]) {
    super(`Validation failed: ${fields.join(', ')}`);
  }
}

// Register strategy
errorRegistry.register(ValidationError, (error) => ({
  status: 422,
  message: error.message,
  fields: error.fields, // Include validation details
}));
```

### Example 2: Add Rate Limiting

```typescript
export class RateLimitExceededError extends Error {
  constructor(public retryAfter: number) {
    super('Too many requests');
  }
}

errorRegistry.register(RateLimitExceededError, (error) => ({
  status: 429,
  message: error.message,
  retryAfter: error.retryAfter,
}));
```

### Example 3: Environment-Specific Strategies

```typescript
// Development: verbose errors
if (process.env.NODE_ENV === 'development') {
  errorRegistry.register(InternalError, (error) => ({
    status: 500,
    message: error.message,
    stack: error.stack, // Include stack trace
  }));
}

// Production: sanitized errors
if (process.env.NODE_ENV === 'production') {
  errorRegistry.register(InternalError, (error) => ({
    status: 500,
    message: 'Internal server error', // Hide details
  }));
}
```

---

## 6. Implementation Checklist

- [ ] Create updated `errorHandler.ts` with Strategy Pattern
- [ ] **Create dedicated unit test file** `tests/api/controllers/errorHandler.test.ts`
- [ ] **Test all strategies independently** (notFoundStrategy, badRequestStrategy)
- [ ] **Test ErrorMappingRegistry** (lookup logic, unknown errors, null/undefined handling)
- [ ] **Test logError function** (both branches: context provided and undefined)
- [ ] **Test handleControllerError** (mapped errors, unmapped errors, fallback)
- [ ] **Verify coverage improvement** (target: 98-100% for errorHandler, 85-90%+ overall)
- [ ] Update existing controller tests if needed
- [ ] Verify all controllers still work correctly
- [ ] Add tests for registry extensibility
- [ ] Document error mapping strategies in code comments
- [ ] Consider adding TypeScript types for ErrorResponse extensions

---

## 7. Comparison Summary

| Aspect | Before (If-Chain) | After (Strategy Pattern) |
|--------|-------------------|--------------------------|
| **Adding new error** | Modify function | Register strategy |
| **Code complexity** | O(n) linear growth | O(1) lookup |
| **Maintainability** | Poor (long if-chain) | Good (declarative map) |
| **Testability** | One function, many branches | Independent strategies |
| **Branch Coverage** | 91.66% (missing branches) | **98-100%** (estimated) |
| **Overall Coverage** | 76.92% branches | **85-90%+** (estimated) |
| **Test Isolation** | Poor (indirect testing) | Excellent (isolated units) |
| **Extensibility** | Limited | High (register custom) |
| **SOLID principles** | Violates Open/Closed | Follows Open/Closed |
| **Code duplication** | High | Low (reuse strategies) |
| **Readability** | Decreases with growth | Consistent |

---

## 8. References

- **Gang of Four**: "Design Patterns: Elements of Reusable Object-Oriented Software" (1994)
  - Strategy Pattern: pp. 315-323
- **SOLID Principles**: Open/Closed Principle (Bertrand Meyer, 1988)
- **Project Context**: Backend error handling for iPalpiti Audio Segmentation API

---

## 9. Conclusion

The Strategy Pattern provides a clean, maintainable solution for error mapping in our REST API. **The primary motivation for applying this pattern is to significantly improve testability and achieve higher code coverage percentages.**

By encapsulating each error mapping as a strategy and managing them through a registry, we achieve:

1. **Extensibility** without modifying existing code
2. **Clarity** through declarative configuration
3. **Testability** through isolated strategies - **Key Benefit**
4. **Higher Coverage** - Projected improvement from 76.92% to 85-90%+ branch coverage
5. **Performance** through efficient Map lookups
6. **Flexibility** for future requirements

### Testability & Coverage Summary

- **Current State**: 76.92% branch coverage, 91.66% for errorHandler (missing critical branches)
- **Target State**: 85-90%+ branch coverage, 98-100% for errorHandler
- **Key Improvement**: Isolated, testable components instead of tightly coupled conditional chains
- **Test Strategy**: Dedicated unit tests for each strategy, registry, and handler component

This is a low-complexity, high-value improvement that sets a strong foundation for error handling as the application grows, with **improved testability and higher code coverage as the primary goals**.

---

**Document created for**: iPalpiti Audio Segmentation Backend  
**Date**: November 2025  
**Location**: `backend/tech-docs/Plan/strategy-pattern-plan.md`

