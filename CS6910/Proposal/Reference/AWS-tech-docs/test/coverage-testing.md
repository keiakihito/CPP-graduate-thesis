# Code Coverage Testing Guide

This document explains how to run tests with code coverage and view the HTML coverage reports for the iPalpiti Audio Segmentation Backend.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Running Tests with Coverage](#running-tests-with-coverage)
4. [Viewing HTML Coverage Reports](#viewing-html-coverage-reports)
5. [Understanding Coverage Metrics](#understanding-coverage-metrics)
6. [Coverage Configuration](#coverage-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Integration Tests Setup](#integration-tests-setup)

---

## Overview

Code coverage measures how much of your source code is executed when tests run. This helps identify:
- **Untested code**: Areas that need more tests
- **Dead code**: Code that's never executed
- **Test quality**: How well your tests exercise the codebase

### Coverage Types

- **Statements**: Percentage of code statements executed
- **Branches**: Percentage of conditional branches (if/else) tested
- **Functions**: Percentage of functions called
- **Lines**: Percentage of code lines executed

---

## Prerequisites

1. **Node.js and npm** installed
2. **Dependencies installed**: Run `npm install` in the `backend` directory
3. **Jest configured**: Coverage is already set up in `jest.config.cjs`

---

## Running Tests with Coverage

### Basic Coverage Command

Run all tests with coverage:

```bash
cd backend
npm run test:coverage
```

This command:
- Runs all test suites (unit + integration)
- Generates coverage reports in multiple formats
- Displays coverage summary in the terminal
- Creates HTML report in `coverage/lcov-report/`

### Coverage with Watch Mode

Watch for file changes and re-run coverage:

```bash
npm run test:coverage:watch
```

Useful during development to see coverage changes in real-time.

### Regular Tests (No Coverage)

Run tests without coverage (faster):

```bash
npm run test
```

### Watch Mode (No Coverage)

Run tests in watch mode without coverage:

```bash
npm run test:watch
```

---

## Viewing HTML Coverage Reports

The HTML coverage report provides an interactive, visual representation of your code coverage.

### Method 1: HTTP Server (Recommended)

Start a local HTTP server to view the coverage report:

```bash
npm run coverage:serve
```

This will:
- Start a server on `http://localhost:8080`
- Display the coverage report in your browser
- Show a message like:

```
📊 Coverage Report Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   URL: http://localhost:8080
   Directory: /path/to/backend/coverage/lcov-report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Press Ctrl+C to stop the server
```

**Steps:**
1. Run `npm run coverage:serve`
2. Open your browser
3. Navigate to `http://localhost:8080`
4. Browse the interactive coverage report
5. Press `Ctrl+C` to stop the server when done

### Method 2: Open Directly (macOS)

Open the HTML file directly in your default browser:

```bash
npm run coverage:open
```

This automatically opens `coverage/lcov-report/index.html` in your browser.

### Method 3: Manual Browser Open

1. Navigate to the coverage directory:
   ```bash
   cd backend/coverage/lcov-report
   ```

2. Open `index.html` in your browser:
   - **macOS**: `open index.html`
   - **Linux**: `xdg-open index.html`
   - **Windows**: Double-click `index.html`

---

## Understanding Coverage Metrics

### Coverage Summary

After running `npm run test:coverage`, you'll see a summary like:

```
=============================== Coverage summary ===============================
Statements   : 91.39% ( 223/244 )
Branches     : 76.92% ( 80/104 )
Functions    : 91.93% ( 57/62 )
Lines        : 92.95% ( 211/227 )
================================================================================
```

**What this means:**
- **Statements (91.39%)**: 223 out of 244 code statements were executed
- **Branches (76.92%)**: 80 out of 104 conditional branches were tested
- **Functions (91.93%)**: 57 out of 62 functions were called
- **Lines (92.95%)**: 211 out of 227 code lines were executed

### HTML Report Features

The HTML report provides:

1. **Overview Dashboard**
   - Overall coverage percentages
   - Coverage by file/directory
   - Color-coded indicators (green = good, red = needs work)

2. **File-by-File Breakdown**
   - Click any file to see detailed coverage
   - Line-by-line highlighting:
     - **Green**: Covered by tests
     - **Red**: Not covered
     - **Yellow**: Partially covered (branches)

3. **Branch Coverage**
   - Shows which `if/else` branches were tested
   - Highlights untested conditional paths

4. **Function Coverage**
   - Lists all functions and their coverage status
   - Shows which functions were never called

### Reading the HTML Report

1. **Main Page**: Shows overall coverage and file list
2. **File View**: Click a file to see:
   - Line numbers with coverage indicators
   - Uncovered lines highlighted in red
   - Branch coverage details
   - Function coverage details

---

## Coverage Configuration

Coverage is configured in `jest.config.cjs`:

```javascript
collectCoverageFrom: [
  "src/**/*.{ts,tsx}",
  "!src/**/*.d.ts",
  "!src/**/index.ts",
  "!src/**/*.test.ts",
],
coverageDirectory: "coverage",
coverageReporters: [
  "text",           // Console output
  "text-summary",   // Summary in console
  "html",           // HTML report (for browser viewing)
  "lcov",           // LCOV format (for CI tools)
  "json",           // JSON format
],
```

### Coverage Thresholds

Currently set to `0` (no enforcement). To enforce minimum coverage:

```javascript
coverageThreshold: {
  global: {
    branches: 80,    // Require 80% branch coverage
    functions: 80,   // Require 80% function coverage
    lines: 80,       // Require 80% line coverage
    statements: 80,  // Require 80% statement coverage
  },
},
```

---

## Troubleshooting

### Issue: Coverage Report Not Generated

**Problem**: Running `npm run test:coverage` but no `coverage/` directory appears.

**Solutions**:
1. Check Jest configuration:
   ```bash
   cat jest.config.cjs | grep coverage
   ```

2. Ensure tests are running:
   ```bash
   npm run test  # Should pass first
   ```

3. Check for errors in test output

### Issue: HTML Report Shows "File not found"

**Problem**: `npm run coverage:serve` shows "Coverage report not found".

**Solution**: Generate coverage first:
```bash
npm run test:coverage
# Wait for completion, then:
npm run coverage:serve
```

### Issue: Port 8080 Already in Use

**Problem**: `coverage:serve` fails because port 8080 is occupied.

**Solution**: Use a different port:
```bash
PORT=3001 node scripts/serve-coverage.mjs
```

### Issue: Coverage Shows 0% for Some Files

**Possible Causes**:
1. **File not included**: Check `collectCoverageFrom` in `jest.config.cjs`
2. **File not tested**: No tests exist for that file
3. **Test not running**: Test file has errors or is skipped

**Solution**: Check test files and ensure they're running:
```bash
npm run test -- --listTests
```

### Issue: Integration Tests Failing

**Problem**: Integration tests fail with `ECONNREFUSED` on port 5435.

**Solution**: Start the test database:
```bash
docker-compose -f docker-compose.test.yml up -d postgres-test
```

See [Integration Tests Setup](#integration-tests-setup) for details.

---

## Integration Tests Setup

Integration tests require a running PostgreSQL database. Coverage will still be generated even if integration tests fail, but you'll get more accurate coverage with all tests passing.

### Quick Setup

1. **Start test database**:
   ```bash
   cd backend
   docker-compose -f docker-compose.test.yml up -d postgres-test
   ```

2. **Wait for database to be ready** (check logs):
   ```bash
   docker-compose -f docker-compose.test.yml logs postgres-test
   ```

3. **Create `.env.test` file** (if not exists):
   ```bash
   cp env.test.example .env.test
   ```

4. **Verify `.env.test` contains**:
   ```env
   TEST_DATABASE_URL=postgres://postgres:test_password@localhost:5435/iPalpiti_test
   ```

5. **Run tests with coverage**:
   ```bash
   npm run test:coverage
   ```

### Verify Database is Running

```bash
# Check if container is running
docker ps | grep postgres-test

# Check if port is listening
lsof -i :5435

# Test connection manually
psql postgres://postgres:test_password@localhost:5435/iPalpiti_test
```

### Stop Test Database

When done testing:
```bash
docker-compose -f docker-compose.test.yml down
```

---

## Coverage Report Files

After running coverage, you'll find these files in `backend/coverage/`:

```
coverage/
├── lcov-report/          # HTML report (browse this!)
│   ├── index.html       # Main dashboard
│   ├── base.css         # Styles
│   ├── block-navigation.js
│   └── [source files]/ # Coverage for each source file
├── lcov.info            # LCOV format (for CI/CD tools)
└── coverage-final.json  # JSON format (for programmatic access)
```

### Using Coverage Reports

- **HTML Report**: Best for human review and exploration
- **LCOV Format**: Used by CI/CD tools (GitHub Actions, GitLab CI, etc.)
- **JSON Format**: For programmatic analysis or custom reporting tools

---

## Best Practices

1. **Run coverage regularly**: Check coverage after adding new features
2. **Aim for high coverage**: Target 80%+ for critical code paths
3. **Focus on quality**: High coverage doesn't mean good tests
4. **Review uncovered code**: Identify why code isn't covered
5. **Use HTML report**: Visual inspection helps find gaps
6. **Set thresholds gradually**: Start low, increase over time

---

## Example Workflow

```bash
# 1. Make code changes
# ... edit files ...

# 2. Run tests with coverage
npm run test:coverage

# 3. Check terminal output for summary
# ... review coverage percentages ...

# 4. View detailed HTML report
npm run coverage:serve
# Open http://localhost:8080 in browser

# 5. Review uncovered lines (red highlights)
# Add tests for uncovered code

# 6. Re-run coverage to verify improvement
npm run test:coverage

# 7. Stop server when done
# Press Ctrl+C in terminal
```

---

## Summary

- **Generate coverage**: `npm run test:coverage`
- **View HTML report**: `npm run coverage:serve` then open `http://localhost:8080`
- **Current coverage**: ~91% statements, ~77% branches, ~92% functions, ~93% lines
- **Integration tests**: Require Docker database running on port 5435
- **Report location**: `backend/coverage/lcov-report/index.html`

For questions or issues, check the [Troubleshooting](#troubleshooting) section or review the Jest configuration in `jest.config.cjs`.

---

**Last Updated**: November 2025  
**Location**: `backend/tech-docs/test/coverage-testing.md`

