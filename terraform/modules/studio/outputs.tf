output "studio_domain_id" {
  description = "SageMaker Studio Domain ID"
  value       = aws_sagemaker_domain.studio_domain.id
}

output "studio_domain_url" {
  description = "SageMaker Studio Domain URL"
  value       = aws_sagemaker_domain.studio_domain.url
}

output "studio_user_profile_name" {
  description = "SageMaker Studio User Profile Name"
  value       = aws_sagemaker_user_profile.studio_user.user_profile_name
}

output "mlflow_tracking_server_arn" {
  description = "MLflow Tracking Server ARN"
  value       = aws_sagemaker_mlflow_tracking_server.studio_mlflow_server.arn
}

output "mlflow_tracking_server_url" {
  description = "MLflow Tracking Server URL"
  value       = aws_sagemaker_mlflow_tracking_server.studio_mlflow_server.tracking_server_url
}

output "studio_execution_role_arn" {
  description = "Studio Execution Role ARN"
  value       = aws_iam_role.studio_execution_role.arn
}

output "studio_logs_group_name" {
  description = "CloudWatch Log Group for Studio"
  value       = aws_cloudwatch_log_group.studio_logs.name
}

# Debug outputs for troubleshooting
output "vpc_id" {
  description = "VPC ID being used"
  value       = data.aws_vpc.default.id
}

output "existing_subnet_ids" {
  description = "Existing subnet IDs found"
  value       = local.existing_subnet_ids
}

output "created_subnet_ids" {
  description = "Created subnet IDs"
  value       = local.created_subnet_ids
}

output "studio_subnet_ids" {
  description = "Final subnet IDs used for Studio"
  value       = local.studio_subnet_ids
} 