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

variable "ec2_instance_type" {
  description = "EC2 instance type for MLflow server"
  type        = string
}

variable "key_pair_name" {
  description = "AWS Key Pair name for EC2 instances"
  type        = string
}

variable "db_instance_class" {
  description = "RDS instance class for MLflow backend store"
  type        = string
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
}

variable "db_username" {
  description = "RDS master username"
  type        = string
} 