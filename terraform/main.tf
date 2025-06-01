terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.1"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Get current AWS account and region
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Random suffix for unique resource names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket for MLflow artifacts and YOLO datasets
resource "aws_s3_bucket" "mlflow_bucket" {
  bucket = "${var.project_name}-artifacts-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "${var.project_name}-artifacts"
    Environment = "dev"
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "mlflow_bucket_versioning" {
  bucket = aws_s3_bucket.mlflow_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_bucket_encryption" {
  bucket = aws_s3_bucket.mlflow_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_bucket_pab" {
  bucket = aws_s3_bucket.mlflow_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Conditional deployment based on deployment_mode
locals {
  is_studio_mode = var.deployment_mode == "studio"
  is_custom_mode = var.deployment_mode == "custom"
}

# Studio Module (for SageMaker Studio deployment)
module "studio" {
  count  = local.is_studio_mode ? 1 : 0
  source = "./modules/studio"

  project_name                    = var.project_name
  aws_region                      = var.aws_region
  bucket_suffix                   = random_string.bucket_suffix.result
  mlflow_bucket_name              = aws_s3_bucket.mlflow_bucket.bucket
  mlflow_bucket_arn               = aws_s3_bucket.mlflow_bucket.arn
  studio_domain_name              = var.studio_domain_name
  enable_studio_code_editor       = var.enable_studio_code_editor
  enable_studio_jupyter_server    = var.enable_studio_jupyter_server
  automatic_model_registration    = var.automatic_model_registration
  weekly_maintenance_window_start = var.weekly_maintenance_window_start
}

# Custom Module (for EC2-based MLflow with RDS)
module "custom" {
  count  = local.is_custom_mode ? 1 : 0
  source = "./modules/custom"

  project_name         = var.project_name
  aws_region           = var.aws_region
  bucket_suffix        = random_string.bucket_suffix.result
  mlflow_bucket_name   = aws_s3_bucket.mlflow_bucket.bucket
  mlflow_bucket_arn    = aws_s3_bucket.mlflow_bucket.arn
  ec2_instance_type    = var.ec2_instance_type
  key_pair_name        = var.key_pair_name
  db_instance_class    = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_username          = var.db_username
} 