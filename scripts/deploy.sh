#!/bin/bash

# MLflow YOLO Platform Deployment Script
# This script deploys the infrastructure for training YOLO models with dual deployment options

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install Terraform first."
        exit 1
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials are not configured. Run 'aws configure' first."
        exit 1
    fi
    
    print_success "All prerequisites are satisfied"
}

# Detect deployment mode
detect_deployment_mode() {
    print_header "Detecting Deployment Mode"
    
    cd terraform
    
    # Check if terraform.tfvars exists
    if [ ! -f "terraform.tfvars" ]; then
        print_warning "No terraform.tfvars found. Using default Studio mode."
        print_status "To customize deployment, copy terraform-studio.tfvars.example or terraform-custom.tfvars.example to terraform.tfvars"
        DEPLOYMENT_MODE="studio"
    else
        # Extract deployment mode from terraform.tfvars
        DEPLOYMENT_MODE=$(grep -E '^deployment_mode\s*=' terraform.tfvars | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "studio")
    fi
    
    print_status "Deployment mode: $DEPLOYMENT_MODE"
    
    # Validate deployment mode
    if [[ "$DEPLOYMENT_MODE" != "studio" && "$DEPLOYMENT_MODE" != "custom" ]]; then
        print_error "Invalid deployment mode: $DEPLOYMENT_MODE. Must be 'studio' or 'custom'."
        exit 1
    fi
    
    # Check Custom mode requirements
    if [ "$DEPLOYMENT_MODE" = "custom" ]; then
        KEY_PAIR_NAME=$(grep -E '^key_pair_name\s*=' terraform.tfvars | sed 's/.*=\s*"\([^"]*\)".*/\1/' || echo "")
        if [ -z "$KEY_PAIR_NAME" ] || [ "$KEY_PAIR_NAME" = "" ]; then
            print_error "Custom deployment mode requires key_pair_name to be set in terraform.tfvars"
            print_error "Create a key pair: aws ec2 create-key-pair --key-name my-key --query 'KeyMaterial' --output text > my-key.pem"
            print_error "Then set: key_pair_name = \"my-key\" in terraform.tfvars"
            exit 1
        fi
        print_status "Using key pair: $KEY_PAIR_NAME"
    fi
    
    cd ..
}

# Initialize Terraform
init_terraform() {
    print_header "Initializing Terraform"
    
    cd terraform
    terraform init
    print_success "Terraform initialized successfully"
    cd ..
}

# Plan infrastructure deployment
plan_deployment() {
    print_header "Planning Infrastructure Deployment"
    
    cd terraform
    terraform plan -out=tfplan
    print_success "Terraform plan created successfully"
    cd ..
}

# Deploy infrastructure
deploy_infrastructure() {
    print_header "Deploying Infrastructure"
    
    cd terraform
    terraform apply tfplan
    print_success "Infrastructure deployed successfully"
    cd ..
}

# Wait for services to be ready
wait_for_services() {
    print_header "Waiting for Services to be Ready"
    
    cd terraform
    
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        wait_for_studio_mlflow
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        wait_for_custom_mlflow
    fi
    
    cd ..
}

# Wait for SageMaker MLflow tracking server to be ready (Studio mode)
wait_for_studio_mlflow() {
    print_status "Waiting for SageMaker MLflow tracking server to be ready..."
    print_warning "This may take 5-10 minutes as the service provisions resources..."
    
    # Get tracking server ARN and region
    STUDIO_MLFLOW_ARN=$(terraform output -raw studio_mlflow_tracking_server_arn 2>/dev/null || echo "")
    AWS_REGION=$(terraform output -raw aws_region)
    
    if [ -z "$STUDIO_MLFLOW_ARN" ]; then
        print_error "Could not get MLflow tracking server ARN from outputs"
        print_status "Available outputs:"
        terraform output | grep -i mlflow || echo "No MLflow outputs found"
        return 1
    fi
    
    print_status "MLflow ARN: $STUDIO_MLFLOW_ARN"
    
    # Extract tracking server name from ARN (format: arn:aws:sagemaker:region:account:mlflow-tracking-server/server-name)
    TRACKING_SERVER_NAME=$(echo "$STUDIO_MLFLOW_ARN" | sed 's/.*mlflow-tracking-server\///')
    
    if [ -z "$TRACKING_SERVER_NAME" ]; then
        print_error "Could not extract tracking server name from ARN: $STUDIO_MLFLOW_ARN"
        return 1
    fi
    
    print_status "Tracking server name: $TRACKING_SERVER_NAME"
    print_status "Region: $AWS_REGION"
    
    max_attempts=30  # 15 minutes total (30 * 30 seconds)
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Checking SageMaker MLflow tracking server... (attempt $attempt/$max_attempts)"
        
        # Check the tracking server status with better error handling
        STATUS=$(aws sagemaker describe-mlflow-tracking-server \
            --tracking-server-name "$TRACKING_SERVER_NAME" \
            --region "$AWS_REGION" \
            --query 'TrackingServerStatus' \
            --output text 2>&1)
        
        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            print_status "AWS CLI error: $STATUS"
            if [[ "$STATUS" == *"ResourceNotFound"* ]]; then
                print_status "Tracking server not found yet - still provisioning..."
            else
                print_warning "Unexpected error checking server status: $STATUS"
            fi
            STATUS="Provisioning"
        fi
        
        print_status "Server status: $STATUS"
        
        case "$STATUS" in
            "Created")
                print_success "SageMaker MLflow tracking server is ready!"
                return 0
                ;;
            "Creating"|"Starting"|"Provisioning")
                print_status "Server is still initializing..."
                ;;
            "CreateFailed"|"UpdateFailed")
                print_error "SageMaker MLflow tracking server creation failed!"
                # Get more details about the failure
                aws sagemaker describe-mlflow-tracking-server \
                    --tracking-server-name "$TRACKING_SERVER_NAME" \
                    --region "$AWS_REGION" \
                    --query 'FailureReason' --output text 2>/dev/null || echo "No failure reason available"
                return 1
                ;;
            *)
                print_status "Server status: $STATUS - continuing to wait..."
                ;;
        esac
        
        sleep 30
        ((attempt++))
    done
    
    print_warning "SageMaker MLflow tracking server is taking longer than expected to initialize."
    print_status "You can check the status manually with:"
    print_status "  aws sagemaker describe-mlflow-tracking-server --tracking-server-name $TRACKING_SERVER_NAME --region $AWS_REGION"
    
    # Get the current status one more time for debugging
    print_status "Final status check:"
    aws sagemaker describe-mlflow-tracking-server \
        --tracking-server-name "$TRACKING_SERVER_NAME" \
        --region "$AWS_REGION" 2>/dev/null || print_status "Could not get final status"
    
    return 0
}

# Wait for Custom MLflow server to be ready (Custom mode)
wait_for_custom_mlflow() {
    print_status "Waiting for Custom EC2 MLflow server to be ready..."
    print_warning "This may take 5-15 minutes for the EC2 instance to start and configure MLflow..."
    
    EC2_PUBLIC_IP=$(terraform output -raw mlflow_server_public_ip 2>/dev/null || echo "")
    
    if [ -z "$EC2_PUBLIC_IP" ]; then
        print_error "Could not get EC2 public IP from outputs"
        return 1
    fi
    
    max_attempts=30  # 15 minutes total (30 * 30 seconds)
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Checking MLflow server on EC2... (attempt $attempt/$max_attempts)"
        
        # Check if MLflow server is responding
        if curl -s --connect-timeout 5 "http://$EC2_PUBLIC_IP:5000/health" > /dev/null 2>&1; then
            print_success "Custom EC2 MLflow server is ready!"
            return 0
        else
            print_status "MLflow server not yet responding - still configuring..."
        fi
        
        sleep 30
        ((attempt++))
    done
    
    print_warning "Custom EC2 MLflow server is taking longer than expected to initialize."
    print_warning "You can check the instance logs: aws logs tail /aws/ec2/mlflow/$(terraform output -raw project_name) --region $(terraform output -raw aws_region)"
    return 0
}

# Save deployment information
save_deployment_info() {
    print_status "Saving deployment information..."
    
    cd terraform
    PROJECT_NAME=$(terraform output -raw project_name)
    DEPLOYMENT_MODE=$(terraform output -raw deployment_mode)
    AWS_REGION=$(terraform output -raw aws_region)
    S3_BUCKET=$(terraform output -raw mlflow_bucket_name)
    
    cat > ../deployment_info.txt << EOF
# MLflow YOLO Platform Deployment Information
# Generated on: $(date)

PROJECT_NAME=$PROJECT_NAME
DEPLOYMENT_MODE=$DEPLOYMENT_MODE
AWS_REGION=$AWS_REGION
S3_BUCKET=$S3_BUCKET

EOF

    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        cat >> ../deployment_info.txt << EOF
# Studio Mode Configuration
STUDIO_DOMAIN_URL=$(terraform output -raw studio_domain_url)
STUDIO_MLFLOW_URL=$(terraform output -raw studio_mlflow_tracking_server_url)
STUDIO_EXECUTION_ROLE=$(terraform output -raw studio_execution_role_arn)

# Usage Instructions:
# 1. Go to AWS Console → SageMaker → Studio
# 2. Launch Studio for your user profile
# 3. Use STUDIO_MLFLOW_URL as the MLflow tracking URI
# 4. Upload datasets to S3_BUCKET
# 5. Use STUDIO_EXECUTION_ROLE for SageMaker training jobs
EOF
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        cat >> ../deployment_info.txt << EOF
# Custom Mode Configuration
MLFLOW_UI_URL=$(terraform output -raw mlflow_ui_url)
EC2_PUBLIC_IP=$(terraform output -raw mlflow_server_public_ip)
EC2_PRIVATE_IP=$(terraform output -raw mlflow_server_private_ip)
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
SSH_COMMAND="ssh -i $KEY_PAIR_NAME.pem ec2-user@$(terraform output -raw mlflow_server_public_ip)"

# Usage Instructions:
# 1. Access MLflow UI at MLFLOW_UI_URL
# 2. SSH to server using SSH_COMMAND
# 3. Use MLFLOW_UI_URL as the MLflow tracking URI
# 4. Upload datasets to S3_BUCKET
# 5. Database credentials are stored in AWS Secrets Manager
EOF
    fi
    
    cd ..
    
    print_success "Deployment information saved to deployment_info.txt"
}

# Setup completion message
show_completion_message() {
    echo
    echo "=========================================="
    print_success "🎉 Deployment completed successfully!"
    echo "=========================================="
    echo
    
    cd terraform
    PROJECT_NAME=$(terraform output -raw project_name)
    DEPLOYMENT_MODE=$(terraform output -raw deployment_mode)
    S3_BUCKET=$(terraform output -raw mlflow_bucket_name)
    
    if [ "$DEPLOYMENT_MODE" = "studio" ]; then
        echo "🏢 SageMaker Studio Infrastructure is ready!"
        echo
        echo "📊 Studio Domain: $(terraform output -raw studio_domain_url)"
        echo "🧠 MLflow Server: $(terraform output -raw studio_mlflow_tracking_server_url)"
        echo "🪣 S3 Bucket: $S3_BUCKET"
        echo
        echo "Access Instructions:"
        echo "1. Go to AWS Console → SageMaker → Studio"
        echo "2. Click on your domain and launch Studio"
        echo "3. Start training with integrated MLflow tracking!"
        echo
        echo "Training in Studio notebooks:"
        echo "  import mlflow"
        echo "  mlflow.set_tracking_uri('$(terraform output -raw studio_mlflow_tracking_server_url)')"
        echo "  mlflow.set_experiment('yolo-training')"
        
    elif [ "$DEPLOYMENT_MODE" = "custom" ]; then
        echo "🖥️ Custom EC2-based MLflow Infrastructure is ready!"
        echo
        echo "📊 MLflow UI: $(terraform output -raw mlflow_ui_url)"
        echo "🖥️ EC2 Instance: $(terraform output -raw mlflow_server_public_ip)"
        echo "🗄️ Database: $(terraform output -raw rds_endpoint)"
        echo "🪣 S3 Bucket: $S3_BUCKET"
        echo
        echo "Access Instructions:"
        echo "1. Open $(terraform output -raw mlflow_ui_url) in your browser"
        echo "2. SSH: ssh -i $KEY_PAIR_NAME.pem ec2-user@$(terraform output -raw mlflow_server_public_ip)"
        echo "3. Start training with MLflow tracking!"
        echo
        echo "Training with remote MLflow:"
        echo "  import mlflow"
        echo "  mlflow.set_tracking_uri('$(terraform output -raw mlflow_ui_url)')"
        echo "  mlflow.set_experiment('yolo-training')"
    fi
    
    echo
    echo "Common next steps:"
    echo "1. Upload your YOLO datasets to s3://$S3_BUCKET/datasets/"
    echo "2. Use the provided MLflow URI in your training scripts"
    echo "3. Launch SageMaker training jobs for scalable training"
    echo
    echo "Useful commands:"
    echo "  - View all outputs: cd terraform && terraform output"
    echo "  - Check connection info: cd terraform && terraform output connection_info"
    echo "  - Destroy infrastructure: cd terraform && terraform destroy"
    echo
    echo "All deployment information is saved in deployment_info.txt"
    
    cd ..
}

# Handle script interruption
trap 'print_error "Deployment interrupted!"; exit 1' INT TERM

# Main deployment function
main() {
    print_header "🚀 MLflow YOLO Platform Deployment"
    
    check_prerequisites
    detect_deployment_mode
    init_terraform
    plan_deployment
    deploy_infrastructure
    save_deployment_info
    wait_for_services
    show_completion_message
}

# Run main function
main "$@" 