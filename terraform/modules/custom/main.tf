# Custom EC2-based MLflow Deployment Module
# This module sets up MLflow on EC2 with RDS backend store

# Data sources
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Generate private key for the key pair
resource "tls_private_key" "mlflow_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Create key pair in the current region
resource "aws_key_pair" "mlflow_key" {
  key_name   = "${var.key_pair_name}-${var.bucket_suffix}"
  public_key = tls_private_key.mlflow_key.public_key_openssh

  tags = {
    Name = "${var.project_name}-key-pair"
    Environment = "dev"
    Project = var.project_name
    Region = var.aws_region
  }
}

# Save private key to local file
resource "local_file" "private_key" {
  content  = tls_private_key.mlflow_key.private_key_pem
  filename = "${path.root}/../${aws_key_pair.mlflow_key.key_name}.pem"
  file_permission = "0600"
}

# Random password for RDS
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Security Groups
resource "aws_security_group" "mlflow_ec2_sg" {
  name_prefix = "${var.project_name}-mlflow-ec2-"
  vpc_id      = data.aws_vpc.default.id

  # MLflow UI
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-mlflow-ec2-sg"
    Environment = "dev"
    Project = var.project_name
  }
}

resource "aws_security_group" "rds_sg" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = data.aws_vpc.default.id

  # MySQL/PostgreSQL
  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [aws_security_group.mlflow_ec2_sg.id]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
    Environment = "dev"
    Project = var.project_name
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "mlflow_db_subnet_group" {
  name       = "${var.project_name}-db-subnet-group-${var.bucket_suffix}"
  subnet_ids = data.aws_subnets.default.ids

  tags = {
    Name = "${var.project_name}-db-subnet-group"
    Environment = "dev"
    Project = var.project_name
  }
}

# RDS MySQL Instance for MLflow Backend Store
resource "aws_db_instance" "mlflow_db" {
  identifier = "${var.project_name}-mlflow-db-${var.bucket_suffix}"
  
  # Database configuration
  engine         = "mysql"
  engine_version = "8.0"
  instance_class = var.db_instance_class
  
  # Storage
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_allocated_storage * 2
  storage_encrypted     = true
  
  # Database credentials
  db_name  = "mlflow"
  username = var.db_username
  password = random_password.db_password.result
  
  # Network & Security
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow_db_subnet_group.name
  publicly_accessible    = false
  
  # Backup & Maintenance
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Deletion protection
  skip_final_snapshot = true
  deletion_protection = false

  tags = {
    Name = "${var.project_name}-mlflow-db"
    Environment = "dev"
    Project = var.project_name
  }
}

# IAM Role for EC2 Instance
resource "aws_iam_role" "ec2_mlflow_role" {
  name = "${var.project_name}-ec2-mlflow-role-${var.bucket_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-ec2-mlflow-role"
    Environment = "dev"
    Project = var.project_name
  }
}

# IAM Policy for S3 and other services
resource "aws_iam_role_policy" "ec2_mlflow_policy" {
  name = "${var.project_name}-ec2-mlflow-policy"
  role = aws_iam_role.ec2_mlflow_role.id

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
          "s3:GetBucketLocation"
        ]
        Resource = [
          var.mlflow_bucket_arn,
          "${var.mlflow_bucket_arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.db_credentials.arn
      }
    ]
  })
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "ec2_mlflow_profile" {
  name = "${var.project_name}-ec2-mlflow-profile-${var.bucket_suffix}"
  role = aws_iam_role.ec2_mlflow_role.name
}

# Store RDS credentials in Secrets Manager
resource "aws_secretsmanager_secret" "db_credentials" {
  name                    = "${var.project_name}-mlflow-db-credentials-${var.bucket_suffix}"
  description             = "MLflow database credentials"
  recovery_window_in_days = 0

  tags = {
    Name = "${var.project_name}-db-credentials"
    Environment = "dev"
    Project = var.project_name
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.mlflow_db.username
    password = random_password.db_password.result
    endpoint = aws_db_instance.mlflow_db.endpoint
    port     = aws_db_instance.mlflow_db.port
    dbname   = aws_db_instance.mlflow_db.db_name
  })
}

# User data script for MLflow setup
locals {
  user_data = templatefile("${path.module}/user_data.sh", {
    project_name                = var.project_name
    mlflow_bucket_name         = var.mlflow_bucket_name
    db_endpoint                = aws_db_instance.mlflow_db.endpoint
    db_port                    = aws_db_instance.mlflow_db.port
    db_name                    = aws_db_instance.mlflow_db.db_name
    db_username                = aws_db_instance.mlflow_db.username
    db_password                = random_password.db_password.result
    aws_region                 = var.aws_region
    secret_arn                 = aws_secretsmanager_secret.db_credentials.arn
    MLFLOW_BACKEND_STORE_URI   = "mysql+pymysql://${aws_db_instance.mlflow_db.username}:${random_password.db_password.result}@${aws_db_instance.mlflow_db.endpoint}/${aws_db_instance.mlflow_db.db_name}"
    MLFLOW_DEFAULT_ARTIFACT_ROOT = "s3://${var.mlflow_bucket_name}/mlflow-artifacts"
  })
}

# EC2 Instance for MLflow
resource "aws_instance" "mlflow_server" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.ec2_instance_type
  key_name              = aws_key_pair.mlflow_key.key_name
  subnet_id             = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.mlflow_ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_mlflow_profile.name
  
  user_data = base64encode(local.user_data)

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  tags = {
    Name = "${var.project_name}-mlflow-server"
    Environment = "dev"
    Project = var.project_name
  }

  depends_on = [
    aws_db_instance.mlflow_db,
    aws_secretsmanager_secret_version.db_credentials
  ]
}

# CloudWatch Log Group for MLflow
resource "aws_cloudwatch_log_group" "mlflow_logs" {
  name              = "/aws/ec2/mlflow/${var.project_name}"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-mlflow-logs"
    Environment = "dev"
    Project = var.project_name
  }
} 