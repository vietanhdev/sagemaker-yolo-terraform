#!/usr/bin/env python3
"""
YOLO Training Script with SageMaker MLflow Integration
This script trains YOLO models and tracks experiments using SageMaker MLflow Tracking Server.
Supports both YOLOv8 and YOLO11 models.
"""

import os
import argparse
import json
import time
from pathlib import Path
import logging

# ML Libraries
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from ultralytics import YOLO

# AWS Libraries
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOSageMakerMLflowTracker:
    """YOLO training with SageMaker MLflow experiment tracking"""
    
    def __init__(self, mlflow_uri, experiment_name="yolo-training"):
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup SageMaker MLflow tracking"""
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"SageMaker MLflow tracking URI: {self.mlflow_uri}")
        logger.info(f"Experiment: {self.experiment_name}")
    
    def log_system_info(self):
        """Log system and environment information"""
        try:
            # Log basic system info
            mlflow.log_param("python_version", f"{sys.version_info.major}.{sys.version_info.minor}")
            mlflow.log_param("platform", platform.system())
            
            # AWS/SageMaker specific info
            if 'SAGEMAKER_JOB_NAME' in os.environ:
                mlflow.log_param("sagemaker_job_name", os.environ.get('SAGEMAKER_JOB_NAME'))
            if 'SAGEMAKER_REGION' in os.environ:
                mlflow.log_param("aws_region", os.environ.get('SAGEMAKER_REGION'))
                
        except Exception as e:
            logger.warning(f"Could not log system info: {e}")
    
    def train_model(self, **kwargs):
        """Train YOLO model with SageMaker MLflow tracking"""
        
        # Extract training parameters
        data_path = kwargs.get('data_path')
        model_size = kwargs.get('model_size', 'yolo11n')
        epochs = kwargs.get('epochs', 10)
        imgsz = kwargs.get('imgsz', 640)
        batch_size = kwargs.get('batch_size', 16)
        device = kwargs.get('device', '0' if torch.cuda.is_available() else 'cpu')
        project_name = kwargs.get('project', 'yolo_runs')
        experiment_id = kwargs.get('experiment_id', int(time.time()))
        
        logger.info(f"Starting YOLO training with model: {model_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Dataset: {data_path}")
        
        with mlflow.start_run(run_name=f"yolo-{model_size}-{experiment_id}") as run:
            # Log parameters
            mlflow.log_param("model_size", model_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("image_size", imgsz)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("device", device)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("experiment_id", experiment_id)
            
            # Log system information
            mlflow.log_param("torch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("cuda_device_count", torch.cuda.device_count())
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
            
            # Log additional system info
            self.log_system_info()
            
            try:
                # Initialize YOLO model
                logger.info(f"Loading YOLO model: {model_size}")
                model = YOLO(f'{model_size}.pt')
                
                # Log model architecture info
                mlflow.log_param("model_yaml", str(model.model))
                
                # Start training
                logger.info("Starting training...")
                results = model.train(
                    data=data_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=device,
                    project=project_name,
                    name=f'{model_size}_exp_{experiment_id}',
                    save=True,
                    plots=True,
                    verbose=True,
                    val=True,
                    save_period=max(1, epochs // 10),  # Save checkpoints periodically
                    cache=False,  # Disable caching for SageMaker compatibility
                    workers=4 if torch.cuda.is_available() else 2,
                    patience=50,  # Early stopping patience
                    cos_lr=True,  # Use cosine learning rate scheduler
                    close_mosaic=10,  # Close mosaic augmentation in last N epochs
                )
                
                # Log training metrics
                if hasattr(results, 'results_dict'):
                    for key, value in results.results_dict.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            mlflow.log_metric(key, value)
                
                # Log final metrics from the results
                if hasattr(results, 'box') and hasattr(results.box, 'map'):
                    mlflow.log_metric("final_map", results.box.map)
                    mlflow.log_metric("final_map50", results.box.map50)
                    mlflow.log_metric("final_map75", results.box.map75)
                
                # Save and log model artifacts
                try:
                    model_path = f"{project_name}/{model_size}_exp_{experiment_id}/weights/best.pt"
                    if os.path.exists(model_path):
                        mlflow.log_artifact(model_path, "model")
                        logger.info(f"Model artifacts logged to MLflow")
                    
                    # Log training plots if they exist
                    plots_dir = f"{project_name}/{model_size}_exp_{experiment_id}"
                    if os.path.exists(plots_dir):
                        for plot_file in Path(plots_dir).glob("*.png"):
                            mlflow.log_artifact(str(plot_file), "plots")
                        logger.info(f"Training plots logged to MLflow")
                        
                except Exception as e:
                    logger.warning(f"Could not log artifacts: {e}")
                
                logger.info("Training completed successfully!")
                return model, results
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                mlflow.log_param("error", str(e))
                raise

def download_s3_dataset(s3_bucket, s3_key, local_path="/tmp/dataset"):
    """Download dataset from S3"""
    logger.info(f"Downloading dataset from s3://{s3_bucket}/{s3_key}")
    
    s3_client = boto3.client('s3')
    
    try:
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        
        # List objects in S3 prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_key)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Skip directories
                    if obj['Key'].endswith('/'):
                        continue
                    
                    # Create local file path
                    local_file = os.path.join(local_path, obj['Key'].replace(s3_key, '').lstrip('/'))
                    local_dir = os.path.dirname(local_file)
                    os.makedirs(local_dir, exist_ok=True)
                    
                    # Download file
                    s3_client.download_file(s3_bucket, obj['Key'], local_file)
        
        logger.info(f"Dataset downloaded to {local_path}")
        return local_path
        
    except ClientError as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLO with SageMaker MLflow tracking')
    
    # MLflow configuration
    parser.add_argument('--mlflow-uri', required=True, 
                       help='SageMaker MLflow tracking server URI')
    parser.add_argument('--experiment-name', default='yolo-training',
                       help='MLflow experiment name')
    
    # Data configuration
    parser.add_argument('--data-path', required=True,
                       help='Path to dataset YAML file or S3 URI')
    parser.add_argument('--s3-bucket', 
                       help='S3 bucket containing dataset')
    parser.add_argument('--s3-dataset-key',
                       help='S3 key/prefix for dataset')
    
    # Training configuration - Updated for YOLO11 support
    parser.add_argument('--model-size', default='yolo11n',
                       choices=[
                           # YOLO11 models (recommended)
                           'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
                           # YOLOv8 models (backward compatibility)
                           'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
                       ],
                       help='YOLO model size (YOLO11 recommended)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--project', default='yolo_runs',
                       help='Project name for saving runs')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting YOLO training with SageMaker MLflow integration")
    logger.info(f"Arguments: {vars(args)}")
    
    # Handle S3 dataset download if needed
    data_path = args.data_path
    if args.s3_bucket and args.s3_dataset_key:
        data_path = download_s3_dataset(args.s3_bucket, args.s3_dataset_key)
        # Update data_path to point to the YAML file
        yaml_files = list(Path(data_path).glob("*.yaml")) + list(Path(data_path).glob("*.yml"))
        if yaml_files:
            data_path = str(yaml_files[0])
            logger.info(f"Using dataset configuration: {data_path}")
    
    # Initialize tracker
    tracker = YOLOSageMakerMLflowTracker(
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name
    )
    
    # Start training
    model, results = tracker.train_model(
        data_path=data_path,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        project=args.project,
        experiment_id=int(time.time())
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"MLflow tracking URI: {args.mlflow_uri}")
    logger.info("Check the SageMaker MLflow UI for detailed results and artifacts")

if __name__ == "__main__":
    import sys
    import platform
    main() 