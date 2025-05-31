# Common outputs
output "project_name" {
  description = "Project name"
  value       = var.project_name
}

output "deployment_mode" {
  description = "Deployment mode (studio or ec2)"
  value       = var.deployment_mode
}

output "aws_region" {
  description = "AWS Region"
  value       = var.aws_region
}

output "mlflow_bucket_name" {
  description = "S3 bucket name for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_bucket.bucket
}

output "mlflow_bucket_arn" {
  description = "S3 bucket ARN for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_bucket.arn
}

# Studio-specific outputs
output "studio_domain_id" {
  description = "SageMaker Studio Domain ID (Studio mode only)"
  value       = var.deployment_mode == "studio" ? module.studio[0].studio_domain_id : null
}

output "studio_domain_url" {
  description = "SageMaker Studio Domain URL (Studio mode only)"
  value       = var.deployment_mode == "studio" ? module.studio[0].studio_domain_url : null
}

output "studio_user_profile_name" {
  description = "SageMaker Studio User Profile Name (Studio mode only)"
  value       = var.deployment_mode == "studio" ? module.studio[0].studio_user_profile_name : null
}

output "studio_mlflow_tracking_server_arn" {
  description = "MLflow Tracking Server ARN (Studio mode only)"
  value       = var.deployment_mode == "studio" ? module.studio[0].mlflow_tracking_server_arn : null
}

output "studio_mlflow_tracking_server_url" {
  description = "MLflow Tracking Server URL (Studio mode only)"
  value       = var.deployment_mode == "studio" ? module.studio[0].mlflow_tracking_server_url : null
}

output "studio_execution_role_arn" {
  description = "Studio Execution Role ARN (Studio mode only)"
  value       = var.deployment_mode == "studio" ? module.studio[0].studio_execution_role_arn : null
}

# EC2-specific outputs
output "mlflow_server_public_ip" {
  description = "Public IP address of the MLflow server (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].mlflow_server_public_ip : null
}

output "mlflow_server_private_ip" {
  description = "Private IP address of the MLflow server (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].mlflow_server_private_ip : null
}

output "mlflow_server_instance_id" {
  description = "Instance ID of the MLflow server (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].mlflow_server_instance_id : null
}

output "mlflow_ui_url" {
  description = "MLflow UI URL (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].mlflow_ui_url : null
}

output "rds_endpoint" {
  description = "RDS instance endpoint (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].rds_endpoint : null
}

output "rds_port" {
  description = "RDS instance port (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].rds_port : null
}

output "database_name" {
  description = "MLflow database name (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].database_name : null
}

output "database_username" {
  description = "MLflow database username (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].database_username : null
}

output "secrets_manager_secret_arn" {
  description = "Secrets Manager secret ARN for database credentials (EC2 mode only)"
  value       = var.deployment_mode == "ec2" ? module.ec2[0].secrets_manager_secret_arn : null
}

# Connection information summary
output "connection_info" {
  description = "Connection information based on deployment mode"
  value = var.deployment_mode == "studio" ? {
    mode = "SageMaker Studio"
    studio_domain_url = var.deployment_mode == "studio" ? module.studio[0].studio_domain_url : null
    mlflow_server_url = var.deployment_mode == "studio" ? module.studio[0].mlflow_tracking_server_url : null
    artifact_store = "s3://${aws_s3_bucket.mlflow_bucket.bucket}/mlflow-artifacts"
    access_instructions = "Go to AWS Console → SageMaker → Studio, then launch Studio for your user profile"
    } : var.deployment_mode == "ec2" ? {
    mode = "EC2 with RDS"
    mlflow_ui_url = var.deployment_mode == "ec2" ? module.ec2[0].mlflow_ui_url : null
    ssh_connection = var.deployment_mode == "ec2" && var.key_pair_name != "" ? "ssh -i ${var.key_pair_name}.pem ec2-user@${module.ec2[0].mlflow_server_public_ip}" : "Key pair not specified"
    database_endpoint = var.deployment_mode == "ec2" ? module.ec2[0].rds_endpoint : null
    artifact_store = "s3://${aws_s3_bucket.mlflow_bucket.bucket}/mlflow-artifacts"
    access_instructions = "Open ${module.ec2[0].mlflow_ui_url} in your browser"
    } : {
    mode = "Unknown"
    artifact_store = "s3://${aws_s3_bucket.mlflow_bucket.bucket}/mlflow-artifacts"
  }
} 