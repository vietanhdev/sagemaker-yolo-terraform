#!/usr/bin/env python3
"""
Shared utilities for SageMaker YOLO training scripts.
Contains common functions used by both Studio and custom training modes.
"""

import os
import yaml
import boto3
from pathlib import Path
from sagemaker.pytorch import PyTorch
from sagemaker import Session
from datetime import datetime


def load_deployment_info(info_file="deployment_info.txt"):
    """Load deployment information from the deployment info file."""
    deployment_info = {}
    
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Deployment info file not found: {info_file}")
    
    with open(info_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                deployment_info[key.strip()] = value.strip()
    
    return deployment_info


def load_config(config_file="config.yaml"):
    """Load training configuration from YAML file."""
    if isinstance(config_file, str):
        config_path = Path(__file__).parent / config_file
    else:
        config_path = config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_deployment_mode(deployment_info, expected_mode):
    """Validate that we're in the expected deployment mode."""
    deployment_mode = deployment_info.get('DEPLOYMENT_MODE', '').lower()
    if deployment_mode != expected_mode.lower():
        raise ValueError(f"This script is for {expected_mode} mode only. Current mode: {deployment_mode}")


def build_training_parameters_studio(config, deployment_info):
    """Build training parameters for Studio mode."""
    
    # Get values from deployment_info or config overrides
    s3_bucket = config['overrides']['s3_bucket'] or deployment_info.get('S3_BUCKET')
    execution_role = config['overrides']['execution_role'] or deployment_info.get('STUDIO_EXECUTION_ROLE')
    mlflow_uri = config['overrides']['mlflow_uri'] or deployment_info.get('STUDIO_MLFLOW_URL')
    aws_region = deployment_info.get('AWS_REGION')
    
    if not all([s3_bucket, execution_role, mlflow_uri, aws_region]):
        missing = []
        if not s3_bucket: missing.append('S3_BUCKET')
        if not execution_role: missing.append('STUDIO_EXECUTION_ROLE')
        if not mlflow_uri: missing.append('STUDIO_MLFLOW_URL')
        if not aws_region: missing.append('AWS_REGION')
        raise ValueError(f"Missing required deployment info: {missing}")
    
    return _build_common_parameters(config, s3_bucket, execution_role, mlflow_uri, aws_region)


def build_training_parameters_custom(config, deployment_info):
    """Build training parameters for custom mode."""
    
    # Get values from deployment_info or config overrides
    s3_bucket = config['overrides']['s3_bucket'] or deployment_info.get('S3_BUCKET')
    execution_role = config['overrides']['execution_role'] or deployment_info.get('CUSTOM_SAGEMAKER_EXECUTION_ROLE')
    mlflow_uri = config['overrides']['mlflow_uri'] or deployment_info.get('MLFLOW_UI_URL')
    aws_region = deployment_info.get('AWS_REGION')
    
    if not all([s3_bucket, execution_role, mlflow_uri, aws_region]):
        missing = []
        if not s3_bucket: missing.append('S3_BUCKET')
        if not execution_role: missing.append('CUSTOM_SAGEMAKER_EXECUTION_ROLE')
        if not mlflow_uri: missing.append('MLFLOW_UI_URL')
        if not aws_region: missing.append('AWS_REGION')
        raise ValueError(f"Missing required deployment info: {missing}")
    
    return _build_common_parameters(config, s3_bucket, execution_role, mlflow_uri, aws_region)


def _build_common_parameters(config, s3_bucket, execution_role, mlflow_uri, aws_region):
    """Build common training parameters shared by both modes."""
    
    # Build S3 paths
    dataset_name = config['data']['dataset_name']
    data_yaml_filename = config['data']['data_yaml_filename']
    s3_dataset_path = f"s3://{s3_bucket}/datasets/{dataset_name}/"
    s3_data_yaml_path = f"s3://{s3_bucket}/datasets/{dataset_name}/{data_yaml_filename}"
    
    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment']['run_name_prefix']}_{timestamp}"
    
    return {
        's3_bucket': s3_bucket,
        'execution_role': execution_role,
        'mlflow_uri': mlflow_uri,
        's3_dataset_path': s3_dataset_path,
        's3_data_yaml_path': s3_data_yaml_path,
        'run_name': run_name,
        'aws_region': aws_region
    }


def create_estimator(config, parameters):
    """Create SageMaker PyTorch estimator with configuration."""
    
    # Create SageMaker session with the correct region
    aws_region = parameters['aws_region']
    print(f"   Using AWS region: {aws_region}")
    
    # Create boto3 session with correct region
    boto_session = boto3.Session(region_name=aws_region)
    sagemaker_session = Session(boto_session=boto_session)
    
    max_run_seconds = config['sagemaker']['max_run_hours'] * 60 * 60  # Convert hours to seconds
    
    # Base estimator parameters
    estimator_params = {
        'entry_point': "yolo_training.py",
        'source_dir': "./scripts",
        'role': parameters['execution_role'],
        'instance_type': config['sagemaker']['instance_type'],
        'instance_count': config['sagemaker']['instance_count'],
        'framework_version': config['sagemaker']['framework_version'],
        'py_version': config['sagemaker']['python_version'],
        'environment': {
            "MLFLOW_TRACKING_URI": parameters['mlflow_uri']
        },
        'hyperparameters': {
            'mlflow-uri': parameters['mlflow_uri'],
            'data-path': parameters['s3_data_yaml_path'],
            's3-bucket': parameters['s3_bucket'],
            's3-dataset-key': f"datasets/{config['data']['dataset_name']}/",
            'model-size': config['model']['size'],
            'epochs': config['model']['epochs'],
            'batch-size': config['model']['batch_size'],
            'imgsz': config['model']['image_size'],
            'experiment-name': config['experiment']['name'],
            'run-name': parameters['run_name']
        },
        'max_run': max_run_seconds,
        'use_spot_instances': config['sagemaker']['use_spot_instances'],
        'sagemaker_session': sagemaker_session  # Use session with correct region
    }
    
    # Add max_wait parameter for spot instances
    if config['sagemaker']['use_spot_instances']:
        # max_wait should be at least equal to max_run, typically 1.5-2x for buffer
        estimator_params['max_wait'] = max_run_seconds * 2  # 2x buffer for spot interruptions
    
    estimator = PyTorch(**estimator_params)
    
    return estimator


def print_training_info(config, parameters, mode="Studio"):
    """Print training configuration information."""
    print(f"🚀 Starting SageMaker {mode} training job...")
    print(f"   Using role: {parameters['execution_role']}")
    print(f"   AWS region: {parameters['aws_region']}")
    print(f"   Instance type: {config['sagemaker']['instance_type']}")
    print(f"   Model: {config['model']['size']}")
    print(f"   Epochs: {config['model']['epochs']}")
    print(f"   Batch size: {config['model']['batch_size']}")
    print(f"   Image size: {config['model']['image_size']}")
    print(f"   MLflow URI: {parameters['mlflow_uri']}")
    print(f"   Experiment: {config['experiment']['name']}")
    print(f"   Run name: {parameters['run_name']}")
    print(f"   Training data: {parameters['s3_dataset_path']}")
    print(f"   Spot instances: {config['sagemaker']['use_spot_instances']}")
    print(f"   Max runtime: {config['sagemaker']['max_run_hours']} hours")
    
    # Show max wait time for spot instances
    if config['sagemaker']['use_spot_instances']:
        max_wait_hours = config['sagemaker']['max_run_hours'] * 2
        print(f"   Max wait time: {max_wait_hours} hours (2x runtime for spot interruptions)")


def run_training(config_file, deployment_mode, script_name="training"):
    """Main training function that can be used by both scripts."""
    # Get the directory containing the calling script
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    # Load configuration files
    print("📋 Loading configuration...")
    config = load_config(script_dir / config_file)
    deployment_info = load_deployment_info(root_dir / "deployment_info.txt")
    
    # Validate deployment mode
    validate_deployment_mode(deployment_info, deployment_mode)
    
    # Build training parameters based on mode
    if deployment_mode.lower() == 'studio':
        parameters = build_training_parameters_studio(config, deployment_info)
    else:  # custom mode
        parameters = build_training_parameters_custom(config, deployment_info)
    
    # Create estimator
    estimator = create_estimator(config, parameters)
    
    # Print training information
    print_training_info(config, parameters, deployment_mode.title())
    
    # Start training
    print(f"\n🎯 Launching {script_name} training job...")
    estimator.fit(parameters['s3_dataset_path'])
    
    print("✅ Training job submitted successfully!")
    
    return estimator 