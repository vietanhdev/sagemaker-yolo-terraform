# AWS Configuration
aws_region = "us-east-1"

# Project Configuration
project_name = "yolo-mlflow"

# Deployment Mode: "studio" for SageMaker Studio, "custom" for custom EC2-based deployment
deployment_mode = "studio"

# Compute Configuration
sagemaker_instance_type = "ml.g4dn.xlarge"  # GPU instance for YOLO training

# Studio Configuration (only used when deployment_mode = "studio")
studio_domain_name = ""  # Leave empty to auto-generate
enable_studio_code_editor = true
enable_studio_jupyter_server = true

# Custom Configuration (only used when deployment_mode = "custom")
ec2_instance_type = "t3.medium"
key_pair_name = ""  # REQUIRED for custom mode - set to your AWS key pair name

# RDS Configuration (only used when deployment_mode = "custom")
db_instance_class = "db.t3.micro"
db_allocated_storage = 20
db_username = "mlflow"

# MLflow Configuration
automatic_model_registration = true
weekly_maintenance_window_start = "Sun:02:00"
