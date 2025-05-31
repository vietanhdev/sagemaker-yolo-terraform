variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "bucket_suffix" {
  description = "Random suffix for unique resource names"
  type        = string
}

variable "mlflow_bucket_name" {
  description = "S3 bucket name for MLflow artifacts"
  type        = string
}

variable "mlflow_bucket_arn" {
  description = "S3 bucket ARN for MLflow artifacts"
  type        = string
}

variable "studio_domain_name" {
  description = "SageMaker Studio domain name"
  type        = string
}

variable "enable_studio_code_editor" {
  description = "Enable Code Editor app in SageMaker Studio"
  type        = bool
}

variable "enable_studio_jupyter_server" {
  description = "Enable Jupyter Server app in SageMaker Studio"
  type        = bool
}

variable "automatic_model_registration" {
  description = "Whether to enable automatic model registration in MLflow"
  type        = bool
}

variable "weekly_maintenance_window_start" {
  description = "Weekly maintenance window start time"
  type        = string
} 