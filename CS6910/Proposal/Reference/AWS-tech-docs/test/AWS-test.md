# AWS API Test Checklist

This document mirrors our Postman flow so we can compare Docker/local validation with the AWS deployment.

## 1. Prerequisites
- `npm ci` completed locally.
- `DATABASE_URL` set to the production/staging Postgres connection string.
- AWS credentials available (`AWS_PROFILE` or default profile) with rights to deploy the Serverless stack.

## 2. Local (Docker) Baseline
1. `docker compose -f docker-compose.test.yml down -v`
2. `docker compose -f docker-compose.test.yml up -d --build`
3. Postman → `http://localhost:3000/api/albums`, `/api/albums/{id}/tracks`, `/api/segment`
4. Confirm responses return seeded data; logs via `docker logs ipalpiti-api-test`.

## 3. Option A – Serverless Offline Against AWS RDS (sanity check)
1. `export DATABASE_URL=postgres://…` (production or staging)
2. Run `npm run dev` to start Serverless Offline.
3. Postman → `http://localhost:3000/api/...` (this uses the Lambda handler locally but proves the RDS credentials/firewall are correct).
4. If responses look good, move on to the full deployment.

## 4. Option B – Deploy to AWS
1. `AWS_PROFILE=YOUR_PROFILE serverless deploy --config serverless.yml --stage prod`
2. Note the API Gateway invoke URL (e.g., `https://xyz.execute-api.us-west-1.amazonaws.com`).

## 5. AWS Verification
1. Postman → `https://<invoke-url>/api/albums`
2. Repeat for tracks and segment endpoints.
3. Monitor CloudWatch logs: `aws logs tail /aws/lambda/iPlalpiti-api-dev --follow`

## 6. Cleanup / Rollback
- Remove stack if it is only for QA: `AWS_PROFILE=YOUR_PROFILE serverless remove --stage prod`
- Or redeploy new builds by repeating Step 4.

Use this checklist to compare responses between Docker (local) and the AWS deployment to validate parity before promoting changes.***
