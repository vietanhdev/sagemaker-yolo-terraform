# Works with both deployment modes
from sagemaker.pytorch import PyTorch

# Get the appropriate role from terraform outputs
# For Studio mode:
# role_arn = "arn:aws:iam::123456789012:role/your-studio-execution-role"  # From: terraform output studio_execution_role_arn
# For Custom mode:
# role_arn = "arn:aws:iam::123456789012:role/your-sagemaker-execution-role"  # From: terraform output custom_sagemaker_execution_role_arn
role_arn = "arn:aws:iam::083919001538:role/yolo-mlflow-sagemaker-execution-role-7zj6tv75"  # From terraform output custom_sagemaker_execution_role_arn

estimator = PyTorch(
    entry_point="yolo_training.py",
    source_dir="./scripts",  # Path to your training scripts
    role=role_arn,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.0",
    py_version="py310",
    environment={
        "MLFLOW_TRACKING_URI": "http://107.21.1.121:5000"
    },
    hyperparameters={
        'mlflow-uri': 'http://107.21.1.121:5000',
        'data-path': 's3://yolo-mlflow-artifacts-7zj6tv75/datasets/beverages/data.yaml',
        's3-bucket': 'yolo-mlflow-artifacts-7zj6tv75',
        's3-dataset-key': 'datasets/beverages/',
        'model-size': 'yolo11s',
        'epochs': 50,
        'batch-size': 16,
        'imgsz': 640,
        'experiment-name': 'custom-yolo-training'
    },
    max_run=24*60*60,  # 24 hours timeout
    use_spot_instances=True,  # 90% cost savings
    max_wait=24*60*60
)

print("🚀 Starting SageMaker training job...")
print(f"   Using role: {role_arn}")
print(f"   MLflow URI: http://107.21.1.121:5000")
print(f"   Training data: s3://yolo-mlflow-artifacts-7zj6tv75/datasets/beverages/")

estimator.fit("s3://yolo-mlflow-artifacts-7zj6tv75/datasets/beverages/")