# Docker Dev Notes

Run the API in Docker against an external Postgres (e.g., AWS RDS).

## Prerequisites
- `.env` (git-ignored) with at least:
  ```
  DATABASE_URL=postgres://<db_user>:<db_password>@<host>:5432/<db_name>?sslmode=no-verify
  # or use sslmode=require and add the RDS CA if you want verification
  ```
- If your DB password has special characters, URL-encode it.
- For strict TLS, download the RDS CA bundle and set `NODE_EXTRA_CA_CERTS=/path/to/rds-combined-ca-bundle.pem`.

## Start
```
docker compose -f docker-compose.dev.yml --env-file .env up -d --build
```
- Exposes the API at `http://localhost:3000`.
- Uses `backend/Dockerfile` to build and run `dist/server.js`.

## Stop
```
docker compose -f docker-compose.dev.yml down
```

## Check
- Logs: `docker logs ipalpiti-api-dev`
- Container: `docker ps | grep ipalpiti-api-dev`

## Endpoints
- GET `http://localhost:3000/api/albums`
- GET `http://localhost:3000/api/albums/{id}/tracks`
- POST `http://localhost:3000/api/segment` (JSON body: `{"trackId":"1","startMs":0,"endMs":2000}`)

Use this to validate the API against RDS from inside Docker before deploying to AWS.
