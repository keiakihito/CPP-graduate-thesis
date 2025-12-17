# API Documentation Guide

This guide explains how to access and use the interactive API documentation for the iPalpiti Audio Segmentation API.

## Overview

The API documentation is provided through **Swagger UI**, an interactive web interface that allows you to:
- Browse all available API endpoints
- View request/response schemas
- Test endpoints directly from your browser
- See example requests and responses
- Understand error response formats

## Accessing the API Documentation

### Local Development Server

When running the API locally, the Swagger UI is available at:

```
http://localhost:3000/api-docs
```

### Docker Development Environment

If using Docker Compose for development:

```bash
cd backend
docker-compose -f docker-compose.dev.yml up -d
```

Then access the documentation at:

```
http://localhost:3000/api-docs
```

### AWS Deployment

After deploying to AWS, the documentation will be available at:

```
https://<your-api-gateway-url>/api-docs
```

Replace `<your-api-gateway-url>` with your actual API Gateway endpoint URL.

## Starting the Server

### Option 1: Direct Node.js (Development)

```bash
cd backend
npm run build
npm start
```

The server will start on port 3000 (or the port specified by the `PORT` environment variable).

### Option 2: Docker Compose (Development with Hot-Reload)

```bash
cd backend
docker-compose -f docker-compose.dev.yml up -d
```

This uses `Dockerfile.dev` which includes hot-reload support via `tsx watch`.

### Option 3: Serverless Offline (Lambda Simulation)

```bash
cd backend
npm run dev
```

This starts the serverless offline plugin, which simulates AWS Lambda locally.

## Using Swagger UI

### 1. Viewing Endpoints

Once you open `http://localhost:3000/api-docs` in your browser, you'll see:

- **API Information**: Title, version, and description at the top
- **Server Selection**: Choose the server URL (defaults to `http://localhost:3000`)
- **Endpoints List**: All available API endpoints organized by path

### 2. Exploring an Endpoint

Click on any endpoint to expand it and see:

- **Description**: What the endpoint does
- **Parameters**: Required and optional parameters (path, query, or body)
- **Request Body Schema**: For POST/PUT requests, the expected JSON structure
- **Response Schemas**: Different response formats for various status codes
- **Example Values**: Sample data for testing

### 3. Testing Endpoints

Swagger UI allows you to test endpoints directly:

1. **Expand an endpoint** by clicking on it
2. **Click "Try it out"** button
3. **Fill in parameters**:
   - Path parameters (e.g., `albumId`, `trackId`)
   - Query parameters (e.g., `page`)
   - Request body (for POST requests)
4. **Click "Execute"** to send the request
5. **View the response**:
   - Response code (200, 400, 404, etc.)
   - Response headers
   - Response body

### 4. Example: Testing the Segment Endpoint

To test the audio segmentation endpoint:

1. Navigate to `POST /api/segment`
2. Click "Try it out"
3. Enter the request body:
   ```json
   {
     "trackId": "1",
     "startMs": 0,
     "endMs": 30000
   }
   ```
4. Click "Execute"
5. The response will show:
   - Status code (200 for success)
   - Content-Type: `audio/wav`
   - The audio file will be downloaded automatically

## Available Endpoints

The API documentation includes the following endpoints:

### Albums

- **GET `/api/albums`** - List all albums (with pagination)
  - Query parameter: `page` (optional, default: 1)

### Tracks

- **GET `/api/albums/{albumId}/tracks`** - List tracks in an album
  - Path parameter: `albumId` (required)

- **GET `/api/tracks/{trackId}`** - Get a specific track
  - Path parameter: `trackId` (required)

### Audio Segmentation

- **POST `/api/segment`** - Download an audio segment
  - Request body:
    ```json
    {
      "trackId": "string",
      "startMs": 0,
      "endMs": 30000
    }
    ```
  - Returns: WAV audio file

## OpenAPI Specification

The OpenAPI specification (JSON format) is available at:

```
http://localhost:3000/api-docs/openapi.json
```

This file can be:
- Imported into API testing tools (Postman, Insomnia, etc.)
- Used to generate client SDKs
- Shared with frontend developers
- Integrated into CI/CD pipelines for API contract testing

### Importing into Postman

1. Open Postman
2. Click "Import"
3. Select "Link" tab
4. Enter: `http://localhost:3000/api-docs/openapi.json`
5. Click "Continue" and "Import"

This will create a Postman collection with all endpoints pre-configured.

## Troubleshooting

### Documentation Not Loading

**Issue**: `http://localhost:3000/api-docs` shows an error or blank page.

**Solutions**:
1. Verify the server is running:
   ```bash
   curl http://localhost:3000/api/albums
   ```

2. Check server logs for errors:
   ```bash
   # If using Docker
   docker logs ipalpiti-api-dev
   
   # If using npm start
   # Check console output
   ```

3. Verify the OpenAPI spec is accessible:
   ```bash
   curl http://localhost:3000/api-docs/openapi.json
   ```

4. Check browser console for JavaScript errors (F12 → Console)

### OpenAPI Spec Not Found

**Issue**: `/api-docs/openapi.json` returns 404.

**Solutions**:
1. Ensure `openapi.json` exists in `backend/src/api/`
2. Rebuild the project:
   ```bash
   cd backend
   npm run build
   ```
3. Verify the file is copied to `dist/api/openapi.json` after build

### CORS Issues

**Issue**: Swagger UI cannot load the OpenAPI spec due to CORS errors.

**Solutions**:
1. Ensure the server is running on the same origin as the Swagger UI
2. For AWS deployments, verify CORS is configured in `serverless.yml`
3. Check browser console for specific CORS error messages

### Endpoints Not Working in Swagger UI

**Issue**: "Try it out" requests fail or return errors.

**Solutions**:
1. Verify the server is running and accessible
2. Check that required environment variables are set (e.g., `DATABASE_URL`)
3. Ensure the database is connected and contains test data
4. Check server logs for detailed error messages
5. Verify the request format matches the schema (check required fields)

## Integration with Development Workflow

### Updating the Documentation

The OpenAPI specification is defined in `backend/src/api/openapi.json`. To update:

1. Edit `backend/src/api/openapi.json`
2. Rebuild the project: `npm run build`
3. Restart the server
4. Refresh the Swagger UI in your browser

### Best Practices

1. **Keep Documentation Updated**: Update `openapi.json` whenever you add or modify endpoints
2. **Use Descriptive Descriptions**: Add clear descriptions for each endpoint and parameter
3. **Include Examples**: Provide example request/response values in the schema
4. **Document Error Responses**: Include all possible error status codes and their meanings
5. **Version Control**: Commit `openapi.json` to version control so it stays in sync with code

## Related Documentation

- **Postman Testing Guide**: See `tech-docs/test/postman-test.md` for detailed Postman setup
- **Docker Development**: See `tech-docs/docker/docker-dev.md` for Docker setup
- **API Implementation**: See `src/api/` for the actual API implementation code

## Technical Details

### Implementation

The Swagger UI is implemented using:
- **Swagger UI CDN**: Uses unpkg.com CDN for Swagger UI assets (no npm package required)
- **OpenAPI 3.0**: The specification follows OpenAPI 3.0 format
- **Custom Handlers**: `backend/src/api/swagger.ts` contains handlers for serving the UI and spec

### File Structure

```
backend/
├── src/
│   └── api/
│       ├── openapi.json      # OpenAPI specification
│       ├── swagger.ts        # Swagger UI handlers
│       └── router.ts         # Route registration
└── dist/
    └── api/
        ├── openapi.json      # Copied during build
        └── swagger.js        # Compiled handler
```

### Build Process

The build script (`package.json`) automatically copies `openapi.json` to `dist/api/`:

```json
{
  "scripts": {
    "build": "rm -rf dist && tsc -p tsconfig.build.json && mkdir -p dist/api && cp src/api/openapi.json dist/api/openapi.json"
  }
}
```

This ensures the OpenAPI spec is available in both development (using `tsx`) and production (using compiled `dist/`) environments.

