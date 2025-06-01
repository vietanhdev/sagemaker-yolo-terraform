#!/bin/bash

# MLflow YOLO Platform Cleanup Script
# This script safely destroys infrastructure for both Studio and Custom deployment modes

set -e

# Prevent AWS CLI from opening pagers
export AWS_PAGER=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# Get deployment info
get_deployment_info() {
    cd terraform
    
    if [ ! -f "terraform.tfstate" ]; then
        print_warning "No Terraform state found. Nothing to destroy."
        cd ..
        exit 0
    fi
    
    DEPLOYMENT_MODE=$(terraform output -raw deployment_mode 2>/dev/null | grep -v "Warning\|╷\|│\|╵" | head -1 || echo "unknown")
    PROJECT_NAME=$(terraform output -raw project_name 2>/dev/null | grep -v "Warning\|╷\|│\|╵" | head -1 || echo "yolo-mlflow")
    
    # Get AWS region or use default
    AWS_REGION_RAW=$(terraform output -raw aws_region 2>/dev/null | grep -v "Warning\|╷\|│\|╵" | head -1)
    if [ -z "$AWS_REGION_RAW" ] || [[ "$AWS_REGION_RAW" =~ ^[[:space:]]*$ ]]; then
        AWS_REGION="us-east-1"
    else
        AWS_REGION="$AWS_REGION_RAW"
    fi
    
    print_status "Mode: $DEPLOYMENT_MODE | Region: $AWS_REGION"
    cd ..
}

# Stop running resources
stop_resources() {
    print_status "Stopping running resources..."
    
    cd terraform
    
    # Stop SageMaker Studio apps
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        STUDIO_DOMAIN_ID=$(terraform output -raw studio_domain_id 2>/dev/null | grep -v "Warning\|╷\|│\|╵" | head -1 || echo "")
        if [ ! -z "$STUDIO_DOMAIN_ID" ]; then
            aws sagemaker list-apps --domain-id-equals "$STUDIO_DOMAIN_ID" --region "$AWS_REGION" \
                --query 'Apps[?Status==`InService`].[DomainId,UserProfileName,AppType,AppName]' \
                --output text --no-cli-pager 2>/dev/null | while read domain_id user_profile app_type app_name; do
                if [ ! -z "$app_name" ]; then
                    aws sagemaker delete-app --domain-id "$domain_id" --user-profile-name "$user_profile" \
                        --app-type "$app_type" --app-name "$app_name" --region "$AWS_REGION" --no-cli-pager || true
                fi
            done
        fi
    fi
    
    # Stop EC2 instances
    if [ "$DEPLOYMENT_MODE" = "custom" ]; then
        EC2_INSTANCE_ID=$(terraform output -raw mlflow_server_instance_id 2>/dev/null | grep -v "Warning\|╷\|│\|╵" | head -1 || echo "")
        if [ ! -z "$EC2_INSTANCE_ID" ]; then
            aws ec2 stop-instances --instance-ids "$EC2_INSTANCE_ID" --region "$AWS_REGION" --no-cli-pager || true
        fi
    fi
    
    # Stop training jobs
    aws sagemaker list-training-jobs --status-equals InProgress --region "$AWS_REGION" \
        --query 'TrainingJobSummaries[].TrainingJobName' --output text --no-cli-pager 2>/dev/null | \
    while read job_name; do
        if [ ! -z "$job_name" ] && [[ "$job_name" == *"$PROJECT_NAME"* ]]; then
            aws sagemaker stop-training-job --training-job-name "$job_name" --region "$AWS_REGION" --no-cli-pager || true
        fi
    done
    
    cd ..
}

# Destroy infrastructure
destroy_all() {
    cd terraform
    
    echo
    print_warning "⚠️  This will permanently destroy ALL infrastructure and data!"
    print_warning "• All MLflow experiments and models"
    print_warning "• All S3 artifacts and data"
    print_warning "• All compute resources"
    echo
    read -p "Type 'destroy-everything' to confirm: " -r
    
    if [[ $REPLY == "destroy-everything" ]]; then
        print_status "Destroying infrastructure..."
        terraform destroy -auto-approve -input=false
        print_success "Infrastructure destroyed!"
    else
        print_warning "Destruction cancelled"
        exit 0
    fi
    
    cd ..
}

# Cleanup local files
cleanup_local() {
    print_status "Cleaning local files..."
    
    rm -f *.pem deployment_info.txt
    rm -f terraform/terraform.tfstate.backup terraform/.terraform.lock.hcl
    
    read -p "Remove Terraform state? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf terraform/.terraform terraform/terraform.tfstate
        print_success "Terraform state removed"
    fi
}

# Main
main() {
    echo "🧹 MLflow YOLO Platform Cleanup"
    echo "================================"
    
    if [ ! -d "terraform" ]; then
        print_error "Terraform directory not found"
        exit 1
    fi
    
    get_deployment_info
    stop_resources
    destroy_all
    cleanup_local
    
    echo
    print_success "🎉 Cleanup completed!"
    print_success "All resources destroyed - no more charges"
}

# Handle interruption
trap 'print_error "Cleanup interrupted!"; exit 1' INT TERM

main "$@" 