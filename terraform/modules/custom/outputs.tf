output "mlflow_server_public_ip" {
  description = "Public IP address of the MLflow server"
  value       = aws_instance.mlflow_server.public_ip
}

output "mlflow_server_private_ip" {
  description = "Private IP address of the MLflow server"
  value       = aws_instance.mlflow_server.private_ip
}

output "mlflow_server_instance_id" {
  description = "Instance ID of the MLflow server"
  value       = aws_instance.mlflow_server.id
}

output "mlflow_ui_url" {
  description = "MLflow UI URL"
  value       = "http://${aws_instance.mlflow_server.public_ip}:5000"
}

output "key_pair_name" {
  description = "Name of the created key pair"
  value       = aws_key_pair.mlflow_key.key_name
}

output "private_key_file" {
  description = "Path to the private key file"
  value       = local_file.private_key.filename
}

output "ssh_command" {
  description = "SSH command to connect to the MLflow server"
  value       = "ssh -i ${local_file.private_key.filename} ec2-user@${aws_instance.mlflow_server.public_ip}"
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow_db.endpoint
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.mlflow_db.port
}

output "database_name" {
  description = "MLflow database name"
  value       = aws_db_instance.mlflow_db.db_name
}

output "database_username" {
  description = "MLflow database username"
  value       = aws_db_instance.mlflow_db.username
}

output "secrets_manager_secret_arn" {
  description = "Secrets Manager secret ARN for database credentials"
  value       = aws_secretsmanager_secret.db_credentials.arn
}

output "mlflow_logs_group_name" {
  description = "CloudWatch Log Group for MLflow"
  value       = aws_cloudwatch_log_group.mlflow_logs.name
}

output "ec2_security_group_id" {
  description = "Security Group ID for EC2 instance"
  value       = aws_security_group.mlflow_ec2_sg.id
}

output "rds_security_group_id" {
  description = "Security Group ID for RDS instance"
  value       = aws_security_group.rds_sg.id
}

output "sagemaker_execution_role_arn" {
  description = "SageMaker Execution Role ARN for training jobs"
  value       = aws_iam_role.sagemaker_execution_role.arn
} 