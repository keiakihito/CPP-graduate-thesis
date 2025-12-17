# AWS Setup Guide

This comprehensive guide covers setting up AWS credentials, IAM configuration, AWS STS (Security Token Service), and connecting your application to AWS RDS (PostgreSQL) and S3.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Account Setup](#aws-account-setup)
3. [IAM User and Credentials](#iam-user-and-credentials)
4. [AWS CLI Configuration](#aws-cli-configuration)
5. [AWS STS (Security Token Service)](#aws-sts-security-token-service)
6. [AWS RDS Setup and Connection](#aws-rds-setup-and-connection)
7. [S3 Bucket Setup and Connection](#s3-bucket-setup-and-connection)
8. [Environment Variables Configuration](#environment-variables-configuration)
9. [Verification and Testing](#verification-and-testing)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- **AWS Account**: An active AWS account with appropriate permissions
- **AWS CLI**: Installed and configured (see [AWS CLI Installation](#aws-cli-installation))
- **Node.js**: v20.x or higher
- **Access to AWS Console**: Ability to create IAM users, RDS instances, and S3 buckets

### AWS CLI Installation

If you don't have AWS CLI installed:

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**
Download and install from: https://awscli.amazonaws.com/AWSCLIV2.msi

Verify installation:
```bash
aws --version
```

---

## AWS Account Setup

### 1. Get Your AWS Account ID

1. Log in to [AWS Console](https://console.aws.amazon.com/)
2. Click on your account name (top right)
3. Your **Account ID** is displayed (12-digit number)
4. **Note this down** - you'll need it for IAM role ARNs

### 2. Select AWS Region

Choose a region and stick with it for consistency. This project uses **us-west-1** by default.

**Common regions:**
- `us-west-1` (N. California) - Used in this project
- `us-west-2` (Oregon)
- `us-east-1` (N. Virginia)
- `ap-northeast-1` (Tokyo)

**Important**: All resources (RDS, S3, Lambda) should be in the **same region** to minimize latency and costs.

---

## IAM User and Credentials

### Step 1: Create IAM User

1. Go to **AWS Console → IAM → Users**
2. Click **"Create user"**
3. Enter username: `ipalpiti-api-user` (or your preferred name)
4. Click **"Next"**

### Step 2: Set Permissions

**Option A: Attach Policies Directly (Quick Setup)**

1. Select **"Attach policies directly"**
2. Search and attach these managed policies:
   - `AmazonRDSFullAccess` (for RDS access)
   - `AmazonS3FullAccess` (for S3 access)
   - `AWSLambda_FullAccess` (for Lambda deployment)
   - `IAMFullAccess` (for IAM role creation)
   - `CloudFormationFullAccess` (for Serverless Framework)

**Option B: Create Custom Policy (Recommended for Production)**

1. Go to **IAM → Policies → Create policy**
2. Click **"JSON"** tab and paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "rds:DescribeDBInstances",
        "rds:Connect",
        "rds:DescribeDBClusters"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::ipalpiti-audio-resource",
        "arn:aws:s3:::ipalpiti-audio-resource/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:*",
        "apigateway:*",
        "cloudformation:*",
        "logs:*",
        "iam:PassRole",
        "iam:CreateRole",
        "iam:AttachRolePolicy"
      ],
      "Resource": "*"
    }
  ]
}
```

3. Name it: `iPalpitiAPIPolicy`
4. Click **"Create policy"**
5. Go back to user creation and attach this custom policy

### Step 3: Create Access Keys

1. After creating the user, click on the username
2. Go to **"Security credentials"** tab
3. Scroll to **"Access keys"** section
4. Click **"Create access key"**
5. Select **"Command Line Interface (CLI)"** as use case
6. Check the confirmation box and click **"Next"**
7. Optionally add a description tag
8. Click **"Create access key"**

### Step 4: Save Credentials

**IMPORTANT**: You'll see the credentials **only once**. Save them securely!

You'll see:
- **Access Key ID**: `AKIAIOSFODNN7EXAMPLE`
- **Secret Access Key**: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

**Save these in a secure location** (password manager, encrypted file, etc.)

---

## AWS CLI Configuration

### Method 1: Interactive Configuration (Recommended)

```bash
aws configure
```

You'll be prompted for:
1. **AWS Access Key ID**: Paste your Access Key ID
2. **AWS Secret Access Key**: Paste your Secret Access Key
3. **Default region name**: `us-west-1` (or your chosen region)
4. **Default output format**: `json`

This creates `~/.aws/credentials` and `~/.aws/config` files.

### Method 2: Named Profile (Multiple AWS Accounts)

If you work with multiple AWS accounts, use named profiles:

```bash
aws configure --profile ipalpiti-dev
```

Enter the same information as above. This creates a profile named `ipalpiti-dev`.

**Using a named profile:**
```bash
export AWS_PROFILE=ipalpiti-dev
# Or in your .env file:
AWS_PROFILE=ipalpiti-dev
```

### Method 3: Environment Variables

You can also set credentials via environment variables:

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=us-west-1
```

**Note**: Environment variables take precedence over `~/.aws/credentials`.

### Verify Configuration

Test your AWS CLI configuration:

```bash
aws sts get-caller-identity
```

You should see output like:
```json
{
    "UserId": "AIDAEXAMPLE",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/ipalpiti-api-user"
}
```

---

## AWS STS (Security Token Service)

AWS STS provides temporary security credentials. It's used for:
- **AssumeRole**: Assuming IAM roles for cross-account access
- **GetSessionToken**: Getting temporary credentials for MFA-protected access
- **AssumeRoleWithWebIdentity**: For OIDC (used in GitHub Actions)

### Understanding STS

**Why use STS?**
- **Security**: Temporary credentials expire automatically
- **Least Privilege**: Assume roles with specific permissions
- **Audit Trail**: All STS calls are logged in CloudTrail

### Common STS Operations

#### 1. Get Current Identity

```bash
aws sts get-caller-identity
```

Returns your current AWS identity (user/role ARN, account ID).

#### 2. Assume Role (Cross-Account Access)

If you need to assume a role:

```bash
aws sts assume-role \
  --role-arn arn:aws:iam::123456789012:role/MyRole \
  --role-session-name my-session
```

This returns temporary credentials (AccessKeyId, SecretAccessKey, SessionToken).

#### 3. Get Session Token (MFA)

For MFA-protected access:

```bash
aws sts get-session-token \
  --serial-number arn:aws:iam::123456789012:mfa/your-mfa-device \
  --token-code 123456
```

### Using STS in Your Application

The AWS SDK automatically uses STS when:
- **Lambda functions** assume execution roles
- **EC2 instances** use instance profiles
- **EKS pods** use service accounts

**For this project**: The Lambda function uses an **execution role** (created by Serverless Framework) that automatically handles STS.

---

## AWS RDS Setup and Connection

### Step 1: Create RDS PostgreSQL Instance

1. Go to **AWS Console → RDS → Databases**
2. Click **"Create database"**
3. Select **"Standard create"**
4. Choose **"PostgreSQL"** engine
5. Select version: **PostgreSQL 15.x** (or latest stable)

**Templates:**
- **Production**: Use "Production" template
- **Development**: Use "Free tier" (if eligible) or "Dev/Test"

**Settings:**
- **DB instance identifier**: `ipalpiti-db-dev`
- **Master username**: `postgres` (or your preferred username)
- **Master password**: Create a strong password (save it securely!)
- **Confirm password**: Re-enter the password

**Instance configuration:**
- **DB instance class**: 
  - Free tier: `db.t3.micro` or `db.t4g.micro`
  - Production: `db.t3.small` or larger
- **Storage**: 
  - **Storage type**: General Purpose SSD (gp3)
  - **Allocated storage**: 20 GB (minimum)

**Connectivity:**
- **VPC**: Use default VPC (or create a custom VPC)
- **Subnet group**: Default (or create custom)
- **Public access**: 
  - **Yes** (for development/testing from local machine)
  - **No** (for production - use VPC endpoints)
- **VPC security group**: Create new or use existing
  - **Security group name**: `ipalpiti-db-sg`
  - **Add inbound rule**: 
    - Type: PostgreSQL
    - Port: 5432
    - Source: Your IP address (for local access) or VPC CIDR (for Lambda)

**Database authentication:**
- **Password authentication** (default)

**Additional configuration:**
- **Initial database name**: `iPalpiti` (optional)
- **Backup retention**: 7 days (production) or 0 days (dev)
- **Enable encryption**: Yes (recommended)

6. Click **"Create database"**

**Wait 5-10 minutes** for the database to be created.

### Step 2: Get RDS Connection Details

1. Go to **RDS → Databases → Your database**
2. Find the **"Connectivity & security"** tab
3. Note the following:
   - **Endpoint**: `ipalpiti-db-dev.xxxxx.us-west-1.rds.amazonaws.com`
   - **Port**: `5432`
   - **Database name**: `postgres` (or the name you specified)
   - **Username**: `postgres` (or your master username)

### Step 3: Configure Security Group

To allow connections from your local machine:

1. Go to **EC2 → Security Groups**
2. Find your RDS security group (`ipalpiti-db-sg`)
3. Click **"Edit inbound rules"**
4. Add rule:
   - **Type**: PostgreSQL
   - **Port**: 5432
   - **Source**: 
     - **My IP** (for temporary local access)
     - **Custom**: `0.0.0.0/0` (NOT recommended for production)
     - **VPC CIDR** (for Lambda access)

5. Click **"Save rules"**

### Step 4: Test Connection Locally

Install PostgreSQL client:

**macOS:**
```bash
brew install postgresql
```

**Linux:**
```bash
sudo apt-get install postgresql-client
```

**Test connection:**
```bash
psql -h ipalpiti-db-dev.xxxxx.us-west-1.rds.amazonaws.com \
     -U postgres \
     -d postgres \
     -p 5432
```

Enter your master password when prompted.

### Step 5: Create Application Database

Connect to RDS and create your database:

```sql
CREATE DATABASE "iPalpiti_dev";
\c iPalpiti_dev

-- Create your schema (use your schema.sql file)
\i /path/to/your/schema.sql
```

Or using psql command line:

```bash
psql -h ipalpiti-db-dev.xxxxx.us-west-1.rds.amazonaws.com \
     -U postgres \
     -d postgres \
     -c 'CREATE DATABASE "iPalpiti_dev";'
```

### Step 6: Configure Lambda to Access RDS

For Lambda to access RDS, ensure:

1. **Lambda is in the same VPC** as RDS (if RDS is not publicly accessible)
2. **Security group allows Lambda** to connect
3. **Lambda execution role** has RDS access permissions

**Serverless Framework** handles this automatically, but you can verify in `serverless.yml`:

```yaml
provider:
  vpc:
    securityGroupIds:
      - sg-xxxxxxxxx  # Lambda security group
    subnetIds:
      - subnet-xxxxxxxxx  # Private subnets
      - subnet-yyyyyyyyy
```

---

## S3 Bucket Setup and Connection

### Step 1: Create S3 Bucket

1. Go to **AWS Console → S3 → Buckets**
2. Click **"Create bucket"**
3. **Bucket name**: `ipalpiti-audio-resource` (must be globally unique)
4. **AWS Region**: `us-west-1` (same as your other resources)

**Object Ownership:**
- **ACLs disabled** (recommended)
- **Bucket owner enforced**

**Block Public Access:**
- **Block all public access**: ✅ Enabled (recommended for private audio files)

**Bucket Versioning:**
- **Enable versioning**: Optional (recommended for production)

**Default encryption:**
- **Enable**: ✅
- **Encryption type**: 
  - **SSE-S3** (AWS-managed keys) - Free
  - **SSE-KMS** (Customer-managed keys) - More control

**Advanced settings:**
- **Object Lock**: Disabled (unless you need compliance features)

5. Click **"Create bucket"**

### Step 2: Configure Bucket Permissions

1. Go to your bucket → **"Permissions"** tab
2. **Bucket policy**: Add policy to allow Lambda access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowLambdaAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:role/iPlalpiti-api-dev-us-west-1-lambdaRole"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ipalpiti-audio-resource",
        "arn:aws:s3:::ipalpiti-audio-resource/*"
      ]
    }
  ]
}
```

**Note**: Replace `123456789012` with your AWS Account ID and update the role ARN after first deployment.

### Step 3: Upload Test Audio Files

1. Go to your bucket → **"Objects"** tab
2. Click **"Upload"**
3. Add your audio files (`.m4a`, `.mp3`, `.wav`, etc.)
4. **Key structure** (recommended):
   ```
   albums/
     album-1/
       track-1.m4a
       track-2.m4a
     album-2/
       track-3.m4a
   ```

5. Click **"Upload"**

### Step 4: Verify S3 Access

Test S3 access from command line:

```bash
# List buckets
aws s3 ls

# List objects in your bucket
aws s3 ls s3://ipalpiti-audio-resource/

# Download a file
aws s3 cp s3://ipalpiti-audio-resource/albums/album-1/track-1.m4a ./test-download.m4a
```

---

## Environment Variables Configuration

### Local Development (.env file)

Create `backend/.env` file (git-ignored):

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:your_password@ipalpiti-db-dev.xxxxx.us-west-1.rds.amazonaws.com:5432/iPalpiti_dev?sslmode=require

# Alternative: Individual database parameters
# DB_HOST=ipalpiti-db-dev.xxxxx.us-west-1.rds.amazonaws.com
# DB_PORT=5432
# DB_NAME=iPalpiti_dev
# DB_USER=postgres
# DB_PASSWORD=your_password

# AWS Configuration
AWS_REGION=us-west-1
AWS_PROFILE=ipalpiti-dev  # Optional: if using named profile

# S3 Configuration
S3_BUCKET=ipalpiti-audio-resource

# Application Configuration
SEGMENT_MAX_DURATION_MS=240000  # 4 minutes
FFMPEG_ENABLED=true
FFMPEG_PATH=ffmpeg
FFMPEG_TIMEOUT_MS=15000

# SSL Configuration (optional override)
# DATABASE_SSL=true  # Force SSL
# DATABASE_SSL=false # Disable SSL (for local testing)
```

### AWS Lambda Environment Variables

For Lambda deployment, set environment variables in `serverless.yml`:

```yaml
provider:
  environment:
    DATABASE_URL: ${env:DATABASE_URL, ''}
    S3_BUCKET: ${env:S3_BUCKET, 'ipalpiti-audio-resource'}
    AWS_REGION: ${env:AWS_REGION, 'us-west-1'}
    SEGMENT_MAX_DURATION_MS: ${env:SEGMENT_MAX_DURATION_MS, '240000'}
```

**For production**, use **AWS Systems Manager Parameter Store** or **Secrets Manager**:

```yaml
provider:
  environment:
    DATABASE_URL: ${ssm:/ipalpiti/prod/database-url}
    S3_BUCKET: ${ssm:/ipalpiti/prod/s3-bucket}
```

**Setting parameters in Parameter Store:**

```bash
aws ssm put-parameter \
  --name "/ipalpiti/prod/database-url" \
  --value "postgresql://..." \
  --type "SecureString" \
  --region us-west-1
```

### Environment Variable Priority

The application reads environment variables in this order:

1. **System environment variables** (highest priority)
2. **`.env` file** (loaded automatically in development)
3. **Default values** in code (lowest priority)

---

## Verification and Testing

### 1. Verify AWS Credentials

```bash
aws sts get-caller-identity
```

Should return your IAM user/role information.

### 2. Verify RDS Connection

**From command line:**
```bash
psql $DATABASE_URL -c "SELECT version();"
```

**From Node.js:**
```bash
cd backend
node -e "
const { Pool } = require('pg');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
pool.query('SELECT version()').then(r => { console.log(r.rows[0]); pool.end(); });
"
```

### 3. Verify S3 Access

```bash
# List bucket
aws s3 ls s3://ipalpiti-audio-resource/

# Test from Node.js
cd backend
node -e "
const { S3Client, ListBucketsCommand } = require('@aws-sdk/client-s3');
const s3 = new S3Client({ region: 'us-west-1' });
s3.send(new ListBucketsCommand({})).then(r => console.log(r.Buckets));
"
```

### 4. Test Local Server

```bash
cd backend
npm install
npm run build
npm start
```

Test endpoints:
```bash
curl http://localhost:3000/api/albums
```

### 5. Test Lambda Deployment

```bash
cd backend
AWS_PROFILE=ipalpiti-dev npx serverless deploy --stage dev
```

After deployment, test the API Gateway URL:
```bash
curl https://xxxxx.execute-api.us-west-1.amazonaws.com/api/albums
```

---

## Troubleshooting

### Common Issues

#### 1. "Access Denied" Errors

**Problem**: AWS CLI or SDK returns "Access Denied"

**Solutions**:
- Verify IAM user has correct permissions
- Check if credentials are expired
- Verify AWS region matches in all configurations
- Check if resource ARNs are correct

```bash
# Verify current identity
aws sts get-caller-identity

# Check IAM user policies
aws iam list-attached-user-policies --user-name ipalpiti-api-user
```

#### 2. RDS Connection Timeout

**Problem**: Cannot connect to RDS from local machine

**Solutions**:
- Verify security group allows your IP address
- Check if RDS is publicly accessible
- Verify endpoint and port are correct
- Check if password is correct
- Ensure SSL mode matches (`sslmode=require` for RDS)

```bash
# Test connection
psql -h <endpoint> -U postgres -d postgres

# Check security group
aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
```

#### 3. S3 "Access Denied" Errors

**Problem**: Cannot read/write to S3 bucket

**Solutions**:
- Verify IAM user has S3 permissions
- Check bucket policy allows your user/role
- Verify bucket name is correct
- Check if object key path is correct

```bash
# Test S3 access
aws s3 ls s3://ipalpiti-audio-resource/

# Check IAM permissions
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::123456789012:user/ipalpiti-api-user \
  --action-names s3:GetObject \
  --resource-arns arn:aws:s3:::ipalpiti-audio-resource/*
```

#### 4. Lambda Cannot Connect to RDS

**Problem**: Lambda function times out when connecting to RDS

**Solutions**:
- Ensure Lambda is in the same VPC as RDS
- Verify security group allows Lambda security group
- Check if RDS subnet group includes Lambda subnets
- Verify Lambda execution role has RDS permissions

```bash
# Check Lambda VPC configuration
aws lambda get-function-configuration --function-name iPlalpiti-api-dev-http

# Check RDS security group
aws rds describe-db-instances --db-instance-identifier ipalpiti-db-dev
```

#### 5. Environment Variables Not Loading

**Problem**: Application doesn't read environment variables

**Solutions**:
- Verify `.env` file exists in `backend/` directory
- Check file format (no spaces around `=`)
- Ensure variables are exported before running commands
- For Lambda, verify `serverless.yml` includes environment variables

```bash
# Check environment variables
env | grep -E "DATABASE_URL|S3_BUCKET|AWS_REGION"

# Test .env loading
cd backend
node -e "require('dotenv').config(); console.log(process.env.DATABASE_URL);"
```

#### 6. SSL/TLS Connection Errors

**Problem**: Database connection fails with SSL errors

**Solutions**:
- For RDS: Use `sslmode=require` in connection string
- For local: Use `sslmode=disable` or `sslmode=prefer`
- Check if RDS CA certificate is needed (download from AWS)

```bash
# Test with SSL
psql "postgresql://user:pass@host:5432/db?sslmode=require"

# Test without SSL (local only)
psql "postgresql://user:pass@host:5432/db?sslmode=disable"
```

### Getting Help

1. **AWS Support**: Check AWS documentation and support forums
2. **CloudWatch Logs**: Check Lambda and RDS logs for detailed errors
3. **AWS CLI**: Use `--debug` flag for verbose output
4. **Serverless Framework**: Use `--verbose` flag for deployment details

```bash
# Enable debug mode
export AWS_SDK_LOAD_CONFIG=1
export DEBUG=*

# Verbose serverless deployment
npx serverless deploy --verbose
```

---

## Security Best Practices

### 1. Credential Management

- ✅ **Never commit credentials to git**
- ✅ **Use AWS Secrets Manager** or **Parameter Store** for production
- ✅ **Rotate access keys** regularly (every 90 days)
- ✅ **Use IAM roles** instead of access keys when possible
- ✅ **Enable MFA** for IAM users with console access

### 2. RDS Security

- ✅ **Use strong passwords** (minimum 12 characters, mixed case, numbers, symbols)
- ✅ **Enable encryption at rest**
- ✅ **Use SSL/TLS** for all connections
- ✅ **Restrict security group** to specific IPs or VPC CIDR
- ✅ **Regular backups** and test restore procedures
- ✅ **Use separate databases** for dev/staging/production

### 3. S3 Security

- ✅ **Block public access** (unless specifically needed)
- ✅ **Enable versioning** for important data
- ✅ **Use bucket policies** to restrict access
- ✅ **Enable encryption** (SSE-S3 or SSE-KMS)
- ✅ **Use IAM roles** for Lambda access (not access keys)
- ✅ **Enable CloudTrail** for audit logging

### 4. Network Security

- ✅ **Use VPC** for production resources
- ✅ **Private subnets** for RDS (not publicly accessible)
- ✅ **Security groups** with least privilege
- ✅ **Network ACLs** for additional layer
- ✅ **VPC endpoints** for S3 (avoid internet gateway)

---

## Quick Reference

### Essential Commands

```bash
# AWS CLI configuration
aws configure
aws configure --profile ipalpiti-dev

# Verify credentials
aws sts get-caller-identity

# Test RDS connection
psql $DATABASE_URL -c "SELECT 1;"

# Test S3 access
aws s3 ls s3://ipalpiti-audio-resource/

# Deploy to AWS
cd backend
AWS_PROFILE=ipalpiti-dev npx serverless deploy --stage dev

# View Lambda logs
aws logs tail /aws/lambda/iPlalpiti-api-dev-http --follow
```

### Important URLs

- **AWS Console**: https://console.aws.amazon.com/
- **IAM Console**: https://console.aws.amazon.com/iam/
- **RDS Console**: https://console.aws.amazon.com/rds/
- **S3 Console**: https://console.aws.amazon.com/s3/
- **CloudWatch Logs**: https://console.aws.amazon.com/cloudwatch/

### Environment Variables Checklist

- [ ] `DATABASE_URL` - PostgreSQL connection string
- [ ] `S3_BUCKET` - S3 bucket name
- [ ] `AWS_REGION` - AWS region (e.g., `us-west-1`)
- [ ] `AWS_PROFILE` - Named profile (optional)
- [ ] `SEGMENT_MAX_DURATION_MS` - Max segment duration (optional)

---

## Next Steps

After completing this setup:

1. **Test all endpoints** using Postman or Swagger UI
2. **Set up CI/CD** using GitHub Actions (see `lambda-ci-setup-guide.md`)
3. **Configure monitoring** with CloudWatch alarms
4. **Set up backups** for RDS
5. **Review security** settings and tighten permissions
6. **Document** your specific configuration for your team

For CI/CD setup, see: `doc/AWS-setup/lambda-ci-setup-guide.md`

