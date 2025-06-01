variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "yolo-mlflow"
}

# Deployment Configuration
variable "deployment_mode" {
  description = "Deployment mode: 'studio' for SageMaker Studio with integrated services, 'custom' for EC2-based MLflow with RDS"
  type        = string
  default     = "studio"
  validation {
    condition     = contains(["studio", "custom"], var.deployment_mode)
    error_message = "Deployment mode must be either 'studio' or 'custom'."
  }
}

# Compute Configuration
variable "sagemaker_instance_type" {
  description = "SageMaker training instance type for YOLO"
  type        = string
  default     = "ml.g4dn.xlarge"  # GPU instance for YOLO training
}

# Studio Configuration
variable "studio_domain_name" {
  description = "SageMaker Studio domain name (leave empty to auto-generate)"
  type        = string
  default     = ""
}

variable "enable_studio_code_editor" {
  description = "Enable Code Editor app in SageMaker Studio"
  type        = bool
  default     = true
}

variable "enable_studio_jupyter_server" {
  description = "Enable Jupyter Server app in SageMaker Studio"
  type        = bool
  default     = true
}

# MLflow Configuration
variable "automatic_model_registration" {
  description = "Whether to enable automatic model registration in MLflow"
  type        = bool
  default     = true
}

variable "weekly_maintenance_window_start" {
  description = "Weekly maintenance window start time (format: ddd:hh:mm)"
  type        = string
  default     = "Sun:02:00"
  validation {
    condition = can(regex("^(Mon|Tue|Wed|Thu|Fri|Sat|Sun):[0-2][0-9]:[0-5][0-9]$", var.weekly_maintenance_window_start))
    error_message = "Weekly maintenance window must be in format 'ddd:hh:mm' (e.g., 'Sun:02:00')."
  }
}

# Custom Configuration (only used when deployment_mode = "custom")
variable "ec2_instance_type" {
  description = "EC2 instance type for MLflow server"
  type        = string
  default     = "t3.medium"
}

variable "key_pair_name" {
  description = "AWS Key Pair name for EC2 instances (required for custom deployment)"
  type        = string
  default     = ""
}

# RDS Configuration (only used when deployment_mode = "custom")
variable "db_instance_class" {
  description = "RDS instance class for MLflow backend store"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_username" {
  description = "RDS master username"
  type        = string
  default     = "mlflow"
} 