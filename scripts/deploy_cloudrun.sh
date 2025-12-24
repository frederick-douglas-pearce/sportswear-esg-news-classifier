#!/bin/bash
# Deploy classifiers to Google Cloud Run
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Docker installed
#   3. GCP project with Cloud Run API enabled
#   4. Set environment variables:
#      - GCP_PROJECT_ID: Your GCP project ID
#      - GCP_REGION: Cloud Run region (default: us-central1)
#
# Usage:
#   ./scripts/deploy_cloudrun.sh fp      # Deploy FP classifier
#   ./scripts/deploy_cloudrun.sh ep      # Deploy EP classifier
#   ./scripts/deploy_cloudrun.sh all     # Deploy all classifiers
#   ./scripts/deploy_cloudrun.sh status  # Check deployment status

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
REPOSITORY="gcr.io"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prereqs() {
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: GCP_PROJECT_ID environment variable not set${NC}"
        echo "Export it with: export GCP_PROJECT_ID=your-project-id"
        exit 1
    fi

    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}Error: gcloud CLI not installed${NC}"
        echo "Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not installed${NC}"
        exit 1
    fi

    # Configure Docker for GCR
    gcloud auth configure-docker --quiet 2>/dev/null || true
}

# Get image tag (use git SHA or 'latest')
get_image_tag() {
    if git rev-parse --short HEAD &> /dev/null; then
        echo "$(git rev-parse --short HEAD)"
    else
        echo "latest"
    fi
}

# Deploy a classifier
deploy_classifier() {
    local classifier=$1
    local memory=$2
    local timeout=$3
    local tag=$(get_image_tag)
    local image="${REPOSITORY}/${PROJECT_ID}/${classifier}-classifier-api"

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Deploying ${classifier^^} Classifier${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Build Docker image
    echo -e "\n${GREEN}Building Docker image...${NC}"
    docker build \
        --build-arg CLASSIFIER_TYPE=${classifier} \
        -t ${image}:${tag} \
        -t ${image}:latest \
        .

    # Push to Container Registry
    echo -e "\n${GREEN}Pushing to Container Registry...${NC}"
    docker push ${image}:${tag}
    docker push ${image}:latest

    # Deploy to Cloud Run
    echo -e "\n${GREEN}Deploying to Cloud Run...${NC}"
    gcloud run deploy ${classifier}-classifier-api \
        --image ${image}:${tag} \
        --region ${REGION} \
        --platform managed \
        --memory ${memory} \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 1 \
        --timeout ${timeout} \
        --concurrency 10 \
        --allow-unauthenticated \
        --set-env-vars="CLASSIFIER_TYPE=${classifier}"

    # Get service URL
    local url=$(gcloud run services describe ${classifier}-classifier-api \
        --region ${REGION} \
        --format 'value(status.url)')

    echo -e "\n${GREEN}Deployment complete!${NC}"
    echo -e "Service URL: ${url}"
    echo -e "Health check: ${url}/health"
    echo -e "API docs: ${url}/docs"
}

# Check deployment status
check_status() {
    echo -e "${YELLOW}Checking Cloud Run deployments...${NC}\n"

    for classifier in fp ep esg; do
        echo -e "${GREEN}${classifier^^} Classifier:${NC}"
        if gcloud run services describe ${classifier}-classifier-api \
            --region ${REGION} \
            --format 'table(status.conditions.type,status.conditions.status,status.conditions.message)' \
            2>/dev/null; then
            local url=$(gcloud run services describe ${classifier}-classifier-api \
                --region ${REGION} \
                --format 'value(status.url)' 2>/dev/null)
            echo "URL: ${url}"
        else
            echo "Not deployed"
        fi
        echo ""
    done
}

# Main
main() {
    local action=$1

    case "$action" in
        fp)
            check_prereqs
            deploy_classifier "fp" "2Gi" "300s"
            ;;
        ep)
            check_prereqs
            deploy_classifier "ep" "512Mi" "60s"
            ;;
        esg)
            check_prereqs
            deploy_classifier "esg" "1Gi" "120s"
            ;;
        all)
            check_prereqs
            deploy_classifier "fp" "2Gi" "300s"
            deploy_classifier "ep" "512Mi" "60s"
            # deploy_classifier "esg" "1Gi" "120s"  # Uncomment when ready
            ;;
        status)
            check_prereqs
            check_status
            ;;
        *)
            echo "Usage: $0 {fp|ep|esg|all|status}"
            echo ""
            echo "Commands:"
            echo "  fp      Deploy FP (False Positive) classifier"
            echo "  ep      Deploy EP (ESG Pre-filter) classifier"
            echo "  esg     Deploy ESG multi-label classifier (not yet implemented)"
            echo "  all     Deploy all available classifiers"
            echo "  status  Check deployment status"
            echo ""
            echo "Environment variables:"
            echo "  GCP_PROJECT_ID  Your Google Cloud project ID (required)"
            echo "  GCP_REGION      Cloud Run region (default: us-central1)"
            exit 1
            ;;
    esac
}

main "$@"
