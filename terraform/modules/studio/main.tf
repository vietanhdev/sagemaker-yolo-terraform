# SageMaker Studio Deployment Module
# This module sets up SageMaker Studio with integrated MLflow and additional services

# Data sources for VPC
data "aws_vpc" "default" {
  default = true
}

# Get availability zones for the region
data "aws_availability_zones" "available" {
  state = "available"
}

# Get all subnets for the default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  
  filter {
    name   = "state"
    values = ["available"]
  }
  
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# Get subnet details to check what we have
data "aws_subnet" "default_subnets" {
  count = length(data.aws_subnets.default.ids)
  id    = data.aws_subnets.default.ids[count.index]
}

# Create subnets if none exist in the default VPC
resource "aws_subnet" "studio_subnet" {
  count = length(data.aws_subnets.default.ids) == 0 ? min(length(data.aws_availability_zones.available.names), 2) : 0
  
  vpc_id            = data.aws_vpc.default.id
  cidr_block        = "172.31.${32 + count.index * 16}.0/20"  # Use 172.31.32.0/20, 172.31.48.0/20 to avoid conflicts
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true  # SageMaker Studio needs internet access
  
  tags = {
    Name = "${var.project_name}-studio-subnet-${count.index + 1}"
    Environment = "dev"
    Project = var.project_name
    Purpose = "SageMaker Studio"
  }
}

# Check for existing Internet Gateway by looking at the main route table
data "aws_route_table" "default_main" {
  vpc_id = data.aws_vpc.default.id
  
  filter {
    name   = "association.main"
    values = ["true"]
  }
}

# Check if there's already internet connectivity (IGW route) in the main route table
locals {
  # Check if default route already exists and points to an IGW
  existing_igw_routes = [
    for route in data.aws_route_table.default_main.routes : route
    if route.cidr_block == "0.0.0.0/0" && startswith(coalesce(route.gateway_id, ""), "igw-")
  ]
  
  has_existing_igw = length(local.existing_igw_routes) > 0
  existing_igw_id = local.has_existing_igw ? local.existing_igw_routes[0].gateway_id : null
}

# Combine existing and created subnets
locals {
  # Only use existing subnets that are confirmed to be in the correct VPC
  existing_subnet_ids = [
    for subnet in data.aws_subnet.default_subnets : subnet.id 
    if subnet.state == "available" && subnet.vpc_id == data.aws_vpc.default.id
  ]
  created_subnet_ids = [for subnet in aws_subnet.studio_subnet : subnet.id]
  
  # Use created subnets if no existing ones, otherwise use existing
  studio_subnet_ids = length(local.existing_subnet_ids) > 0 ? local.existing_subnet_ids : local.created_subnet_ids
}

# Get current AWS caller identity
data "aws_caller_identity" "current" {}

# Enhanced SageMaker IAM Role for Studio
resource "aws_iam_role" "studio_execution_role" {
  name = "${var.project_name}-studio-execution-role-${var.bucket_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "sagemaker.amazonaws.com",
            "lambda.amazonaws.com"
          ]
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-studio-execution-role"
    Environment = "dev"
    Project = var.project_name
  }
}

# Comprehensive Studio policy
resource "aws_iam_role_policy" "studio_comprehensive_policy" {
  name = "${var.project_name}-studio-comprehensive-policy"
  role = aws_iam_role.studio_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:CreateBucket"
        ]
        Resource = [
          var.mlflow_bucket_arn,
          "${var.mlflow_bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker-mlflow:*",
          "sagemaker:*",
          "ecr:*",
          "logs:*",
          "cloudwatch:*"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_name}-*"
      }
    ]
  })
}

# Attach managed policies
resource "aws_iam_role_policy_attachment" "studio_execution_policy" {
  role       = aws_iam_role.studio_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "studio_s3_policy" {
  role       = aws_iam_role.studio_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "studio_ecr_policy" {
  role       = aws_iam_role.studio_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}

# SageMaker Studio Domain with enhanced configuration
resource "aws_sagemaker_domain" "studio_domain" {
  domain_name = var.studio_domain_name != "" ? var.studio_domain_name : "${var.project_name}-studio"
  auth_mode   = "IAM"
  vpc_id      = data.aws_vpc.default.id
  subnet_ids  = local.studio_subnet_ids

  # Ensure we wait for any created network resources
  depends_on = [
    aws_subnet.studio_subnet,
    aws_internet_gateway.studio_igw,
    aws_route.internet_access
  ]

  default_user_settings {
    execution_role = aws_iam_role.studio_execution_role.arn
    
    # Enable various Studio apps
    default_landing_uri = "studio::"
    
    studio_web_portal = "ENABLED"
    
    # Jupyter Server App settings
    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = "system"
        sagemaker_image_arn = "arn:aws:sagemaker:${var.aws_region}:081325390199:image/jupyter-server-3"
      }
      
      lifecycle_config_arns = var.enable_studio_jupyter_server ? [] : []
    }
    
    # Code Editor App settings
    code_editor_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
      
      lifecycle_config_arns = var.enable_studio_code_editor ? [] : []
    }
    
    # Kernel Gateway App settings for notebooks
    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
        sagemaker_image_arn = "arn:aws:sagemaker:${var.aws_region}:081325390199:image/datascience-1.0"
      }
    }
  }

  tags = {
    Name = "${var.project_name}-studio-domain"
    Environment = "dev"
    Project = var.project_name
  }
}

# SageMaker User Profile
resource "aws_sagemaker_user_profile" "studio_user" {
  domain_id         = aws_sagemaker_domain.studio_domain.id
  user_profile_name = "${var.project_name}-user"
  
  user_settings {
    execution_role = aws_iam_role.studio_execution_role.arn
    
    # Enable specific apps
    studio_web_portal = "ENABLED"
    
    # Override default settings if needed
    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = "system"
      }
    }
    
    code_editor_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }
  }

  tags = {
    Name = "${var.project_name}-studio-user"
    Environment = "dev"
    Project = var.project_name
  }
}

# SageMaker MLflow Tracking Server (Enhanced for Studio)
resource "aws_sagemaker_mlflow_tracking_server" "studio_mlflow_server" {
  tracking_server_name = "${var.project_name}-studio-mlflow-${var.bucket_suffix}"
  artifact_store_uri   = "s3://${var.mlflow_bucket_name}/mlflow-artifacts"
  role_arn            = aws_iam_role.studio_execution_role.arn
  
  automatic_model_registration = var.automatic_model_registration
  weekly_maintenance_window_start = var.weekly_maintenance_window_start
  
  tags = {
    Name        = "${var.project_name}-studio-mlflow-server"
    Environment = "dev"
    Project     = var.project_name
    DeploymentMode = "studio"
  }
}

# CloudWatch Log Group for Studio
resource "aws_cloudwatch_log_group" "studio_logs" {
  name              = "/aws/sagemaker/studio/${var.project_name}"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-studio-logs"
    Environment = "dev"
    Project = var.project_name
  }
}

# Create Internet Gateway only if none exists and we created subnets
resource "aws_internet_gateway" "studio_igw" {
  count = length(aws_subnet.studio_subnet) > 0 && !local.has_existing_igw ? 1 : 0
  
  vpc_id = data.aws_vpc.default.id
  
  tags = {
    Name = "${var.project_name}-studio-igw"
    Environment = "dev"
    Project = var.project_name
  }
}

# Add internet gateway route to main route table only if needed
resource "aws_route" "internet_access" {
  count = length(aws_subnet.studio_subnet) > 0 && !local.has_existing_igw ? 1 : 0
  
  route_table_id         = data.aws_route_table.default_main.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.studio_igw[0].id
} 