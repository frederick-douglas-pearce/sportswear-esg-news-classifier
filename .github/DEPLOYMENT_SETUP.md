# CI/CD Deployment Setup

This guide explains how to configure GitHub Actions for automatic deployment to Google Cloud Run.

## Required GitHub Secrets

Navigate to your repository's Settings > Secrets and variables > Actions and add the following secrets:

### 1. GCP_PROJECT_ID (Required)

Your Google Cloud project ID.

```
Example: my-esg-classifier-project
```

### 2. GCP_SA_KEY (Required)

A service account JSON key with the following roles:
- **Cloud Run Admin** - Deploy and manage Cloud Run services
- **Storage Admin** - Push images to Container Registry
- **Service Account User** - Act as the service account

**To create the service account:**

```bash
# Set your project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Create service account
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Deployer"

# Grant required roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=github-actions@${PROJECT_ID}.iam.gserviceaccount.com

# The contents of key.json should be added as the GCP_SA_KEY secret
cat key.json
```

**Important:** Delete the local key file after adding it to GitHub secrets:
```bash
rm key.json
```

### 3. GCP_REGION (Optional)

The Google Cloud region for Cloud Run deployment. Defaults to `us-central1`.

```
Examples: us-central1, us-east1, europe-west1
```

## Deployment Triggers

The workflow deploys via manual trigger or automation:

1. **Manual trigger** - Use "Run workflow" button in Actions tab:
   - `fp` - Deploy only FP classifier
   - `ep` - Deploy only EP classifier
   - `all` - Deploy both classifiers

2. **Automated via retrain.py** - After major/minor version promotion:
   ```bash
   # Retrain and auto-promote if metrics improve (triggers deployment)
   uv run python scripts/retrain.py --classifier fp --auto-promote
   ```
   The script calls `gh workflow run deploy.yml` for major/minor versions.

3. **Patch versions skipped** - Only major/minor versions trigger redeployment.
   Patch versions update model files but don't require container rebuilds.

## Deployed Services

After deployment, services are available at:

| Classifier | Cloud Run Service | Memory | Timeout |
|------------|-------------------|--------|---------|
| FP | `fp-classifier-api` | 2GB | 300s |
| EP | `ep-classifier-api` | 512MB | 60s |

## Verifying Deployment

Check deployment status:

```bash
# List deployed services
gcloud run services list --region us-central1

# Get service URL
gcloud run services describe fp-classifier-api --region us-central1 --format 'value(status.url)'

# Test health endpoint
curl $(gcloud run services describe fp-classifier-api --region us-central1 --format 'value(status.url)')/health
```

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure the service account has all required roles
2. **Image not found**: Check that Container Registry API is enabled
3. **Build failures**: Check Docker build logs in the Actions tab

### Enable Required APIs

```bash
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com
```

## Local Testing

Before deploying, test locally with Docker:

```bash
# Build FP classifier
docker build --build-arg CLASSIFIER_TYPE=fp -t fp-classifier-api .

# Run locally
docker run -p 8000:8000 -e CLASSIFIER_TYPE=fp fp-classifier-api

# Test health endpoint
curl http://localhost:8000/health
```
