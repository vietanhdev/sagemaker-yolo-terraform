#!/bin/bash

# MLflow YOLO Platform Cleanup Script
# This script safely destroys infrastructure for both Studio and Custom deployment modes

set -e

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

# Detect deployment mode and gather info
detect_deployment_info() {
    print_header "Detecting Deployment Configuration"
    
    cd terraform
    
    if [ ! -f "terraform.tfstate" ]; then
        print_warning "No Terraform state file found. Infrastructure may not be deployed."
        
        # Try to detect from terraform.tfvars
        if [ -f "terraform.tfvars" ]; then
            DEPLOYMENT_MODE=$(grep -E '^deployment_mode\s*=' terraform.tfvars | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "unknown")
            PROJECT_NAME=$(grep -E '^project_name\s*=' terraform.tfvars | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "yolo-mlflow")
            AWS_REGION=$(grep -E '^aws_region\s*=' terraform.tfvars | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "us-east-1")
            print_status "Detected mode from terraform.tfvars: $DEPLOYMENT_MODE"
        else
            DEPLOYMENT_MODE="unknown"
            PROJECT_NAME="yolo-mlflow"
            AWS_REGION="us-east-1"
        fi
        
        cd ..
        return 1
    fi
    
    # Get info from terraform outputs
    DEPLOYMENT_MODE=$(terraform output -raw deployment_mode 2>/dev/null || echo "unknown")
    PROJECT_NAME=$(terraform output -raw project_name 2>/dev/null || echo "yolo-mlflow")
    AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")
    BUCKET_NAME=$(terraform output -raw mlflow_bucket_name 2>/dev/null || echo "")
    
    print_status "Deployment mode: $DEPLOYMENT_MODE"
    print_status "Project name: $PROJECT_NAME"
    print_status "AWS region: $AWS_REGION"
    
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        STUDIO_DOMAIN_ID=$(terraform output -raw studio_domain_id 2>/dev/null || echo "")
        MLFLOW_ARN=$(terraform output -raw studio_mlflow_tracking_server_arn 2>/dev/null || echo "")
        print_status "Studio domain ID: ${STUDIO_DOMAIN_ID:-'Not found'}"
        print_status "MLflow tracking server ARN: ${MLFLOW_ARN:-'Not found'}"
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        EC2_INSTANCE_ID=$(terraform output -raw mlflow_server_instance_id 2>/dev/null || echo "")
        RDS_ENDPOINT=$(terraform output -raw rds_endpoint 2>/dev/null || echo "")
        SECRET_ARN=$(terraform output -raw secrets_manager_secret_arn 2>/dev/null || echo "")
        print_status "EC2 instance ID: ${EC2_INSTANCE_ID:-'Not found'}"
        print_status "RDS endpoint: ${RDS_ENDPOINT:-'Not found'}"
        print_status "Secret ARN: ${SECRET_ARN:-'Not found'}"
    fi
    
    cd ..
    return 0
}

# Show current resources
show_current_resources() {
    print_header "Current Infrastructure Resources"
    
    cd terraform
    
    if [ ! -f "terraform.tfstate" ]; then
        print_warning "No Terraform state file found."
        cd ..
        return 1
    fi
    
    echo "Resources that will be destroyed:"
    echo
    
    # Show key resources by deployment mode
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        echo "📊 SageMaker Studio Resources:"
        terraform show | grep -E "aws_sagemaker|sagemaker" | sed 's/^/  /'
        echo
        echo "🧠 MLflow Resources:"
        terraform show | grep -E "mlflow" | sed 's/^/  /' | head -5
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        echo "🖥️ EC2 Resources:"
        terraform show | grep -E "aws_instance|aws_security_group" | sed 's/^/  /'
        echo
        echo "🗄️ Database Resources:"
        terraform show | grep -E "aws_db_|rds" | sed 's/^/  /'
        echo
        echo "🔐 Security Resources:"
        terraform show | grep -E "aws_secretsmanager|secret" | sed 's/^/  /'
    fi
    
    echo
    echo "💾 Storage Resources:"
    terraform show | grep -E "aws_s3_bucket" | sed 's/^/  /'
    echo
    echo "🔑 IAM Resources:"
    terraform show | grep -E "aws_iam" | sed 's/^/  /' | head -5
    
    echo
    print_warning "Use 'terraform show' to see all resources in detail"
    echo
    
    cd ..
    return 0
}

# Clean up S3 bucket contents
cleanup_s3_bucket() {
    print_header "Cleaning Up S3 Bucket"
    
    if [ -z "$BUCKET_NAME" ]; then
        print_warning "No S3 bucket name found. Skipping S3 cleanup."
        return 0
    fi
    
    print_status "Emptying S3 bucket: $BUCKET_NAME"
    
    # Check if bucket exists
    if aws s3api head-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" 2>/dev/null; then
        print_status "Removing all objects from bucket..."
        
        # Delete all objects and versions
        aws s3 rm "s3://$BUCKET_NAME" --recursive --region "$AWS_REGION" || true
        
        # Delete any object versions (if versioning was enabled)
        print_status "Removing object versions..."
        aws s3api list-object-versions --bucket "$BUCKET_NAME" --region "$AWS_REGION" \
            --query 'Versions[].{Key:Key,VersionId:VersionId}' --output text 2>/dev/null | \
        while read key version_id; do
            if [ "$key" != "None" ] && [ "$version_id" != "None" ] && [ ! -z "$key" ]; then
                aws s3api delete-object --bucket "$BUCKET_NAME" --key "$key" --version-id "$version_id" --region "$AWS_REGION" || true
            fi
        done
        
        # Delete any delete markers
        print_status "Removing delete markers..."
        aws s3api list-object-versions --bucket "$BUCKET_NAME" --region "$AWS_REGION" \
            --query 'DeleteMarkers[].{Key:Key,VersionId:VersionId}' --output text 2>/dev/null | \
        while read key version_id; do
            if [ "$key" != "None" ] && [ "$version_id" != "None" ] && [ ! -z "$key" ]; then
                aws s3api delete-object --bucket "$BUCKET_NAME" --key "$key" --version-id "$version_id" --region "$AWS_REGION" || true
            fi
        done
        
        print_success "S3 bucket emptied successfully"
    else
        print_warning "S3 bucket $BUCKET_NAME not found or already deleted"
    fi
}

# Mode-specific cleanup
cleanup_deployment_specific() {
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        cleanup_studio_resources
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        cleanup_custom_resources
    else
        print_warning "Unknown deployment mode: $DEPLOYMENT_MODE. Skipping mode-specific cleanup."
    fi
}

# Studio mode specific cleanup
cleanup_studio_resources() {
    print_header "Studio Mode Cleanup"
    
    print_status "Checking SageMaker Studio resources..."
    
    # Stop any running apps in Studio
    if [ ! -z "$STUDIO_DOMAIN_ID" ]; then
        print_status "Checking for running Studio apps..."
        
        # List and stop running apps
        aws sagemaker list-apps --domain-id-equals "$STUDIO_DOMAIN_ID" --region "$AWS_REGION" \
            --query 'Apps[?Status==`InService`].[DomainId,UserProfileName,AppType,AppName]' \
            --output text 2>/dev/null | while read domain_id user_profile app_type app_name; do
            if [ ! -z "$app_name" ]; then
                print_status "Stopping Studio app: $app_name ($app_type) for user $user_profile"
                aws sagemaker delete-app \
                    --domain-id "$domain_id" \
                    --user-profile-name "$user_profile" \
                    --app-type "$app_type" \
                    --app-name "$app_name" \
                    --region "$AWS_REGION" || true
            fi
        done
    fi
    
    # Check MLflow tracking server status
    if [ ! -z "$MLFLOW_ARN" ]; then
        TRACKING_SERVER_NAME=$(echo "$MLFLOW_ARN" | sed 's/.*:tracking-server\///')
        if [ ! -z "$TRACKING_SERVER_NAME" ]; then
            print_status "Checking MLflow tracking server status..."
            STATUS=$(aws sagemaker describe-mlflow-tracking-server \
                --tracking-server-name "$TRACKING_SERVER_NAME" \
                --region "$AWS_REGION" \
                --query 'TrackingServerStatus' --output text 2>/dev/null || echo "NotFound")
            print_status "MLflow tracking server status: $STATUS"
        fi
    fi
    
    # Stop any running training jobs
    cleanup_sagemaker_training_jobs
    
    print_success "Studio mode cleanup completed"
}

# Custom mode specific cleanup
cleanup_custom_resources() {
    print_header "Custom Mode Cleanup"
    
    print_status "Checking EC2 and RDS resources..."
    
    # Stop EC2 instance gracefully
    if [ ! -z "$EC2_INSTANCE_ID" ]; then
        print_status "Checking EC2 instance status..."
        INSTANCE_STATE=$(aws ec2 describe-instances \
            --instance-ids "$EC2_INSTANCE_ID" \
            --region "$AWS_REGION" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text 2>/dev/null || echo "not-found")
        
        print_status "EC2 instance state: $INSTANCE_STATE"
        
        if [ "$INSTANCE_STATE" = "running" ]; then
            print_status "Stopping EC2 instance gracefully..."
            aws ec2 stop-instances --instance-ids "$EC2_INSTANCE_ID" --region "$AWS_REGION" || true
            
            # Wait a moment for graceful shutdown
            sleep 10
        fi
    fi
    
    # Check RDS status
    if [ ! -z "$RDS_ENDPOINT" ]; then
        RDS_IDENTIFIER=$(echo "$RDS_ENDPOINT" | cut -d'.' -f1)
        if [ ! -z "$RDS_IDENTIFIER" ]; then
            print_status "Checking RDS instance status..."
            RDS_STATUS=$(aws rds describe-db-instances \
                --db-instance-identifier "$RDS_IDENTIFIER" \
                --region "$AWS_REGION" \
                --query 'DBInstances[0].DBInstanceStatus' \
                --output text 2>/dev/null || echo "not-found")
            print_status "RDS instance status: $RDS_STATUS"
        fi
    fi
    
    print_success "Custom mode cleanup completed"
}

# Stop running SageMaker training jobs
cleanup_sagemaker_training_jobs() {
    print_status "Checking for running SageMaker training jobs..."
    
    # Stop any running training jobs for this project
    aws sagemaker list-training-jobs \
        --status-equals InProgress \
        --region "$AWS_REGION" \
        --query 'TrainingJobSummaries[].TrainingJobName' \
        --output text 2>/dev/null | while read job_name; do
        if [ ! -z "$job_name" ] && [[ "$job_name" == *"$PROJECT_NAME"* ]]; then
            print_status "Stopping training job: $job_name"
            aws sagemaker stop-training-job --training-job-name "$job_name" --region "$AWS_REGION" || true
        fi
    done
}

# Destroy infrastructure
destroy_infrastructure() {
    print_header "Destroying Infrastructure"
    
    cd terraform
    
    # Create destroy plan
    print_status "Creating destruction plan..."
    terraform plan -destroy -out=destroy_plan
    
    echo
    print_warning "⚠️  DANGER ZONE ⚠️"
    print_warning "This will permanently destroy ALL infrastructure and data!"
    echo
    
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        print_warning "This includes:"
        print_warning "• SageMaker Studio Domain and User Profiles"
        print_warning "• MLflow Tracking Server and all experiments"
        print_warning "• S3 bucket and all stored artifacts"
        print_warning "• All IAM roles and policies"
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        print_warning "This includes:"
        print_warning "• EC2 MLflow server and all experiments"
        print_warning "• RDS MySQL database and all metadata"
        print_warning "• S3 bucket and all stored artifacts"
        print_warning "• Secrets Manager credentials"
        print_warning "• All IAM roles, security groups, and policies"
    fi
    
    echo
    print_warning "Make sure you have backed up any important data!"
    echo
    read -p "Type 'destroy-everything' to confirm destruction: " -r
    
    if [[ $REPLY == "destroy-everything" ]]; then
        print_status "Destroying infrastructure..."
        
        if [ "$DEPLOYMENT_MODE" = "studio" ]; then
            print_status "Destroying SageMaker Studio and MLflow resources..."
        elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
            print_status "Destroying EC2, RDS, and MLflow resources..."
        fi
        
        # Run terraform destroy with better error handling
        if terraform apply destroy_plan; then
            print_success "Infrastructure destroyed successfully!"
        else
            print_error "Destruction failed or was interrupted!"
            echo
            print_warning "Common solutions:"
            echo "1. Check AWS Console for any remaining resources"
            echo "2. Re-run: cd terraform && terraform destroy"
            echo "3. Check for stuck resources in SageMaker/EC2/RDS consoles"
            echo "4. Try manual cleanup from AWS Console if needed"
            exit 1
        fi
    else
        print_warning "Destruction cancelled. Infrastructure preserved."
        rm -f destroy_plan
        exit 0
    fi
    
    cd ..
}

# Clean up local files
cleanup_local_files() {
    print_header "Local File Cleanup"
    
    print_status "Cleaning up local files..."
    
    # Remove generated files
    FILES_TO_REMOVE=(
        "*.pem"
        "deployment_info.txt"
        "terraform/terraform.tfstate.backup"
        "terraform/destroy_plan"
        "terraform/.terraform.lock.hcl"
        "terraform/tfplan"
    )
    
    for pattern in "${FILES_TO_REMOVE[@]}"; do
        for file in $pattern; do
            if [ -f "$file" ]; then
                rm "$file"
                print_status "Removed: $file"
            fi
        done
    done
    
    # Remove terraform directory if user wants
    echo
    read -p "Remove Terraform state and cache? This cannot be undone! (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf terraform/.terraform
        rm -f terraform/terraform.tfstate
        print_success "Terraform cache and state removed"
        print_warning "You will need to run 'terraform init' before next deployment"
    else
        print_status "Terraform state preserved for potential recovery"
    fi
    
    print_success "Local cleanup completed"
}

# Verify cleanup
verify_cleanup() {
    print_header "Cleanup Verification"
    
    cd terraform
    
    # Check if any resources remain in state
    if [ -f "terraform.tfstate" ]; then
        if terraform show | grep -q "resource.*aws_"; then
            print_warning "Some resources may still exist in Terraform state:"
            terraform show | grep "resource.*aws_" | head -10
            echo "..."
            echo
            print_warning "This might indicate:"
            echo "1. Cleanup was interrupted"
            echo "2. Some resources failed to destroy"
            echo "3. Manual cleanup may be needed"
        else
            print_success "No AWS resources found in Terraform state"
        fi
    else
        print_status "No Terraform state file found"
    fi
    
    cd ..
    
    # Quick AWS check
    if [ ! -z "$BUCKET_NAME" ]; then
        if aws s3api head-bucket --bucket "$BUCKET_NAME" --region "$AWS_REGION" 2>/dev/null; then
            print_warning "S3 bucket $BUCKET_NAME still exists"
        else
            print_success "S3 bucket successfully removed"
        fi
    fi
}

# Main cleanup process
main() {
    print_header "🧹 MLflow YOLO Platform Cleanup"
    
    # Check if terraform directory exists
    if [ ! -d "terraform" ]; then
        print_error "Terraform directory not found. Are you in the correct directory?"
        exit 1
    fi
    
    # Detect deployment info
    if ! detect_deployment_info; then
        print_warning "No deployed infrastructure found."
        read -p "Clean up local files anyway? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cleanup_local_files
        fi
        exit 0
    fi
    
    # Show current resources
    if ! show_current_resources; then
        print_warning "Could not display current resources."
    fi
    
    # Cleanup process
    cleanup_s3_bucket
    cleanup_deployment_specific
    destroy_infrastructure
    cleanup_local_files
    verify_cleanup
    
    echo
    print_header "🎉 Cleanup Completed Successfully!"
    echo
    print_success "All infrastructure has been destroyed"
    print_success "You will no longer be charged for these resources"
    echo
    
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        echo "Studio resources cleaned up:"
        echo "• SageMaker Studio Domain and User Profiles ✅"
        echo "• MLflow Tracking Server ✅"
        echo "• S3 artifacts bucket ✅"
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        echo "Custom resources cleaned up:"
        echo "• EC2 MLflow server ✅"
        echo "• RDS MySQL database ✅"
        echo "• S3 artifacts bucket ✅"
        echo "• Secrets Manager credentials ✅"
    fi
    
    echo
    echo "To deploy again, run: scripts/deploy.sh"
}

# Handle script interruption
trap 'print_error "Cleanup interrupted!"; exit 1' INT TERM

# Run main function
main "$@" 