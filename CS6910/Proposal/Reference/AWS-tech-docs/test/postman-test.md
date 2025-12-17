# Postman API Testing Guide

This document provides instructions for testing the iPalpiti Audio Segmentation API using Postman.

## Prerequisites

1. **Local Testing**: Ensure you have the backend running locally with `serverless offline`
2. **Docker Testing**: Use Docker containers for isolated API and database testing (recommended)
3. **AWS Testing**: Deploy the API to AWS and obtain the API Gateway URL
4. **Database**: Ensure your database is configured and contains test data

## Environment Variables

Set up the following environment variables in Postman:

- `baseUrl`: 
  - Local (serverless offline): `http://localhost:3000`
  - Docker: `http://localhost:3000`
  - AWS: `https://<your-api-gateway-url>.execute-api.us-west-1.amazonaws.com`

## API Endpoints

### UC-01: Browse Albums

**GET** `{{baseUrl}}/api/albums`

**Query Parameters:**
- `page` (optional): Page number (default: 1)

**Example Request:**
```
GET {{baseUrl}}/api/albums?page=1
```

**Example Response:**
```json
{
  "albums": [
    {
      "id": "1",
      "name": "Album Name",
      "conductor": "Conductor Name",
      "artists": ["Artist 1", "Artist 2"],
      "releaseDate": "2024-01-01",
      "s3Uri": "s3://bucket/album"
    }
  ]
}
```

**Status Codes:**
- `200`: Success
- `500`: Internal server error

---

### UC-02: View Tracks in Album

**GET** `{{baseUrl}}/api/albums/:albumId/tracks`

**Path Parameters:**
- `albumId`: Album ID (required)

**Example Request:**
```
GET {{baseUrl}}/api/albums/1/tracks
```

**Example Response:**
```json
{
  "tracks": [
    {
      "id": "1",
      "albumId": "1",
      "title": "Track Title",
      "durationMs": 180000,
      "s3Key": "s3://bucket/track.wav"
    }
  ]
}
```

**Status Codes:**
- `200`: Success
- `400`: Album id missing from path
- `404`: Album not found or tracks not found
- `500`: Internal server error

---

### UC-03: Preview Track

**GET** `{{baseUrl}}/api/tracks/:trackId`

**Path Parameters:**
- `trackId`: Track ID (required)

**Example Request:**
```
GET {{baseUrl}}/api/tracks/1
```

**Example Response:**
```json
{
  "track": {
    "id": "1",
    "albumId": "1",
    "title": "Track Title",
    "durationMs": 180000,
    "s3Key": "s3://bucket/track.wav"
  }
}
```

**Status Codes:**
- `200`: Success
- `400`: Track id missing from path
- `404`: Track not found
- `500`: Internal server error

---

### UC-04: Define Segment (Download Segment)

**POST** `{{baseUrl}}/api/segment`

**Request Body:**
```json
{
  "trackId": "1",
  "startMs": 0,
  "endMs": 30000
}
```

**Body Parameters:**
- `trackId` (required): Track ID
- `startMs` (required): Start time in milliseconds (must be a number)
- `endMs` (required): End time in milliseconds (must be a number)

**Example Request:**
```
POST {{baseUrl}}/api/segment
Content-Type: application/json

{
  "trackId": "1",
  "startMs": 0,
  "endMs": 30000
}
```

**Example Response:**
```json
{
  "filename": "1-0-30000.wav",
  "contentType": "audio/wav"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request (missing parameters, invalid selection)
- `404`: Track not found
- `500`: Internal server error

---

## Postman Collection Setup

### 1. Create Environment

Create a new environment in Postman with:
- Variable: `baseUrl`
- Initial Value: `http://localhost:3000` (or your AWS URL)
- Current Value: `http://localhost:3000` (or your AWS URL)

### 2. Create Collection

Create a collection named "iPalpiti API" with the following requests:

1. **UC-01: Browse Albums**
   - Method: GET
   - URL: `{{baseUrl}}/api/albums?page=1`

2. **UC-02: View Tracks**
   - Method: GET
   - URL: `{{baseUrl}}/api/albums/1/tracks`
   - Replace `1` with an actual album ID from your database

3. **UC-03: Preview Track**
   - Method: GET
   - URL: `{{baseUrl}}/api/tracks/1`
   - Replace `1` with an actual track ID from your database

4. **UC-04: Define Segment**
   - Method: POST
   - URL: `{{baseUrl}}/api/segment`
   - Headers: `Content-Type: application/json`
   - Body (raw JSON):
     ```json
     {
       "trackId": "1",
       "startMs": 0,
       "endMs": 30000
     }
     ```

## Testing Workflow

### Recommended Testing Order

1. **Start with UC-01**: Get list of albums to find valid album IDs
2. **Use UC-02**: Get tracks for an album to find valid track IDs
3. **Use UC-03**: Get individual track details to verify track exists
4. **Use UC-04**: Create a segment using a valid track ID and time range

### Example Testing Sequence

1. `GET {{baseUrl}}/api/albums?page=1`
   - Note the `id` of the first album (e.g., `1`)

2. `GET {{baseUrl}}/api/albums/1/tracks`
   - Note the `id` of the first track (e.g., `1`)
   - Note the `durationMs` value (e.g., `180000` = 3 minutes)

3. `GET {{baseUrl}}/api/tracks/1`
   - Verify track details match

4. `POST {{baseUrl}}/api/segment`
   - Use track ID from step 2
   - Use `startMs: 0` and `endMs: 30000` (30 seconds, must be less than track duration)

## Error Testing

### Test Invalid Requests

1. **Missing Parameters**
   - `GET {{baseUrl}}/api/albums/:albumId/tracks` with empty `albumId`
   - `POST {{baseUrl}}/api/segment` with missing `trackId`, `startMs`, or `endMs`

2. **Invalid IDs**
   - `GET {{baseUrl}}/api/albums/99999/tracks` (non-existent album)
   - `GET {{baseUrl}}/api/tracks/99999` (non-existent track)

3. **Invalid Segment Selection**
   - `POST {{baseUrl}}/api/segment` with `endMs` greater than track duration
   - `POST {{baseUrl}}/api/segment` with `startMs >= endMs`

## Local Testing Setup

### Option 1: Serverless Offline (Lambda Simulation)

Start the local server using Serverless Framework's offline plugin:

```bash
cd backend
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
npx serverless offline
```

The API will be available at `http://localhost:3000`

### Verify Server is Running

Test with:
```
GET http://localhost:3000/api/albums
```

---

## Docker Container Testing (Recommended)

Docker provides an isolated environment with both the API server and PostgreSQL database running in containers. This is the recommended approach for local testing as it closely matches the production environment.

### Prerequisites

- Docker and Docker Compose installed on your system
- No local PostgreSQL or API server running on ports 3000 or 5435

### Start Docker Containers

The project includes `docker-compose.test.yml` which sets up:
- **PostgreSQL Database**: Containerized database with auto-initialized schema
- **API Server**: Containerized Node.js server running `server.ts`

#### Start Services

```bash
cd backend
docker-compose -f docker-compose.test.yml up -d
```

This command:
1. Builds the API Docker image from `Dockerfile`
2. Starts PostgreSQL container (`ipalpiti-postgres-test`) on port 5435
3. Starts API container (`ipalpiti-api-test`) on port 3000
4. Auto-initializes the database schema from `tests/database/schema.sql`
5. Waits for database to be healthy before starting API

#### Verify Containers are Running

```bash
docker-compose -f docker-compose.test.yml ps
```

You should see both containers in "Up" status.

#### View Logs

```bash
# View all logs
docker-compose -f docker-compose.test.yml logs -f

# View only API logs
docker-compose -f docker-compose.test.yml logs -f api

# View only database logs
docker-compose -f docker-compose.test.yml logs -f postgres-test
```

### Database Connection Details

When using Docker containers, the API connects to the database using:
- **Host**: `postgres-test` (container name, not `localhost`)
- **Port**: `5432` (internal container port)
- **Database**: `iPalpiti_test`
- **User**: `postgres`
- **Password**: `test_password`
- **Connection String**: `postgres://postgres:test_password@postgres-test:5432/iPalpiti_test`

The API container automatically uses this connection string via the `DATABASE_URL` environment variable set in `docker-compose.test.yml`.

### Test with Postman

1. **Set Postman Environment Variable**:
   - Variable: `baseUrl`
   - Value: `http://localhost:3000`

2. **Test Endpoints**:
   - All API endpoints are available at `http://localhost:3000/api/*`
   - The API server (`server.ts`) converts HTTP requests to Lambda event format internally

### Stop Docker Containers

```bash
# Stop containers (keeps data)
docker-compose -f docker-compose.test.yml stop

# Stop and remove containers (keeps volumes/data)
docker-compose -f docker-compose.test.yml down

# Stop and remove containers + volumes (deletes all data)
docker-compose -f docker-compose.test.yml down -v
```

### Rebuild After Code Changes

If you modify the code, rebuild the API container:

```bash
docker-compose -f docker-compose.test.yml up -d --build
```

### Access Database Directly (Optional)

To connect to the database container directly for debugging:

```bash
# Using psql
docker exec -it ipalpiti-postgres-test psql -U postgres -d iPalpiti_test

# Or using docker-compose
docker-compose -f docker-compose.test.yml exec postgres-test psql -U postgres -d iPalpiti_test
```

### Troubleshooting Docker Setup

1. **Port Already in Use**
   - Ensure port 3000 and 5435 are not in use
   - Check: `lsof -i :3000` and `lsof -i :5435`
   - Stop any conflicting services

2. **Container Won't Start**
   - Check logs: `docker-compose -f docker-compose.test.yml logs`
   - Verify Docker is running: `docker ps`
   - Try rebuilding: `docker-compose -f docker-compose.test.yml up -d --build`

3. **Database Connection Errors**
   - Wait for database health check to pass (check logs)
   - Verify database container is healthy: `docker-compose -f docker-compose.test.yml ps`
   - Check API logs for connection errors

4. **Schema Not Initialized**
   - Remove volumes and restart: `docker-compose -f docker-compose.test.yml down -v && docker-compose -f docker-compose.test.yml up -d`
   - Verify schema file exists: `ls backend/tests/database/schema.sql`

### Advantages of Docker Testing

- **Isolated Environment**: No conflicts with local PostgreSQL or Node.js installations
- **Consistent Setup**: Same environment across all developers
- **Easy Cleanup**: Remove containers to start fresh
- **Production-like**: API runs in container similar to deployment
- **Auto-initialization**: Database schema automatically set up

---

## AWS Deployment Testing

### Deploy to AWS

```bash
cd backend
export DATABASE_URL="<your-aws-database-url>"
npx serverless deploy
```

### Get API Gateway URL

After deployment, note the API Gateway URL from the output:
```
endpoints:
  ANY - https://xxxxx.execute-api.us-west-1.amazonaws.com/api/{proxy+}
```

Update your Postman `baseUrl` environment variable to this URL.

## Troubleshooting

### Common Issues

1. **Connection Refused (Local)**
   - Ensure `serverless offline` is running
   - Check if port 3000 is available

2. **404 Not Found**
   - Verify the endpoint URL is correct
   - Check if the resource ID exists in the database

3. **500 Internal Server Error**
   - Check server logs for detailed error messages
   - Verify database connection is configured correctly
   - Ensure `DATABASE_URL` environment variable is set

4. **CORS Errors**
   - Verify CORS is enabled in `serverless.yml`
   - Check if the request origin is allowed

## Notes

- All endpoints require valid database connections
- Track IDs and Album IDs must exist in the database
- Segment selection must be within track duration bounds
- Pagination defaults to page 1 if not specified
- Page size is fixed at 20 items per page

