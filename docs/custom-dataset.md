# Custom Dataset Training with YOLO11

This guide walks you through training YOLO11 models on your custom dataset using the SageMaker MLflow platform.

## 📋 Prerequisites

- Deployed SageMaker MLflow infrastructure
- Custom dataset with annotations
- Basic understanding of YOLO object detection

## 🗂️ Dataset Preparation

### Supported Dataset Formats

YOLO11 supports multiple dataset formats:

1. **YOLO Format** (Recommended)
2. **COCO Format**
3. **Pascal VOC Format**
4. **Custom YAML Format**

### YOLO Format Structure

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/ (optional)
│       ├── image1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── test/ (optional)
│       ├── image1.txt
│       └── ...
└── dataset.yaml
```

### Label Format

Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```

Where:
- `class_id`: Integer class ID (0-indexed)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized width and height (0-1)

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.1 0.15
```

### Dataset YAML Configuration

Create `dataset.yaml`:

```yaml
# Dataset configuration for YOLO11
path: /path/to/dataset  # Root directory
train: images/train     # Relative path to training images
val: images/val         # Relative path to validation images
test: images/test       # Optional: relative path to test images

# Number of classes
nc: 3

# Class names
names:
  0: person
  1: car
  2: bicycle

# Optional: Download script
download: |
  # Add download commands if needed
  echo "Dataset already prepared"
```

## 🔄 Data Conversion Tools

### Converting from COCO Format

```python
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco

# Convert COCO dataset to YOLO format
convert_coco(
    labels_dir='path/to/coco/annotations',
    save_dir='path/to/yolo/dataset',
    use_segments=False,  # Set True for segmentation
    use_keypoints=False,  # Set True for pose estimation
    cls91to80=True  # Convert 91 COCO classes to 80
)
```

### Converting from Pascal VOC Format

```python
from ultralytics.data.converter import convert_dota

# Convert Pascal VOC to YOLO
def convert_voc_to_yolo(voc_dir, output_dir):
    import xml.etree.ElementTree as ET
    import os
    from pathlib import Path
    
    # Implementation for VOC to YOLO conversion
    # (Full script available in examples/)
    pass
```

## 📤 Upload Dataset to S3

### Using AWS CLI

```bash
# Compress your dataset
tar -czf my_dataset.tar.gz my_dataset/

# Upload to S3
aws s3 cp my_dataset.tar.gz s3://YOUR_BUCKET/datasets/

# Upload individual files (alternative)
aws s3 sync my_dataset/ s3://YOUR_BUCKET/datasets/my_dataset/
```

### Using Python

```python
import boto3
import tarfile
from pathlib import Path

def upload_dataset_to_s3(dataset_path, bucket_name, s3_prefix):
    """Upload dataset to S3"""
    s3_client = boto3.client('s3')
    
    # Create compressed archive
    archive_path = f"{dataset_path}.tar.gz"
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(dataset_path, arcname=Path(dataset_path).name)
    
    # Upload to S3
    s3_key = f"{s3_prefix}/{Path(archive_path).name}"
    s3_client.upload_file(archive_path, bucket_name, s3_key)
    
    print(f"Dataset uploaded to s3://{bucket_name}/{s3_key}")
    return s3_key

# Usage
s3_key = upload_dataset_to_s3(
    dataset_path="my_dataset",
    bucket_name="your-bucket-name",
    s3_prefix="datasets"
)
```

## 🚀 Training with Custom Dataset

### Option 1: Local Training

```python
#!/usr/bin/env python3
"""Custom dataset training example"""

from ultralytics import YOLO
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("http://YOUR_MLFLOW_IP:5000")
mlflow.set_experiment("custom-dataset-training")

with mlflow.start_run(run_name="yolo11s-custom"):
    # Load YOLO11 model
    model = YOLO('yolo11s.pt')
    
    # Train on custom dataset
    results = model.train(
        data='path/to/your/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,  # GPU
        project='custom_runs',
        name='yolo11s_custom',
        
        # Training hyperparameters
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=10
    )
    
    # Log metrics to MLflow
    mlflow.log_params({
        "model": "yolo11s",
        "epochs": 100,
        "batch_size": 16,
        "image_size": 640
    })
    
    # Log final metrics
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
```

### Option 2: SageMaker Training

Update your training script to handle custom datasets:

```python
# In sagemaker_train.py
def main():
    # ... existing setup ...
    
    # Custom dataset configuration
    dataset_name = os.environ.get('DATASET_NAME', 'custom_dataset')
    num_classes = int(os.environ.get('NUM_CLASSES', '3'))
    class_names = os.environ.get('CLASS_NAMES', 'person,car,bicycle').split(',')
    
    # Extract and prepare dataset
    dataset_file = os.path.join(input_data_path, f'{dataset_name}.tar.gz')
    if os.path.exists(dataset_file):
        with tarfile.open(dataset_file, 'r:gz') as tar:
            tar.extractall('/opt/ml/input')
    
    # Create dataset YAML
    dataset_yaml = f"""
path: /opt/ml/input/{dataset_name}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {dict(enumerate(class_names))}
"""
    
    with open(f'/opt/ml/input/{dataset_name}.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    # Train model
    model, results = trainer.train_model(
        data_path=f'/opt/ml/input/{dataset_name}.yaml',
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        device='auto',
        project='/opt/ml/model',
        experiment_id=int(time.time())
    )
```

### SageMaker Estimator Configuration

```python
from sagemaker.pytorch import PyTorch

# Custom dataset training on SageMaker
estimator = PyTorch(
    entry_point='sagemaker_train.py',
    source_dir='./scripts',
    role=SAGEMAKER_ROLE,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'MLFLOW_URI': MLFLOW_TRACKING_URI,
        'EXPERIMENT_NAME': 'custom-dataset-yolo11',
        'MODEL_SIZE': 'yolo11s',
        'EPOCHS': 150,
        'BATCH_SIZE': 16,
        'IMGSZ': 640,
        'DATASET_NAME': 'my_custom_dataset',
        'NUM_CLASSES': 5,
        'CLASS_NAMES': 'person,car,truck,bicycle,motorcycle'
    },
    use_spot_instances=True,
    max_wait=10800,  # 3 hours
    max_run=7200,    # 2 hours
    output_path=f's3://{S3_BUCKET}/custom-model-output/',
    volume_size=50  # Larger volume for custom datasets
)

# Start training
training_input = f's3://{S3_BUCKET}/datasets/my_custom_dataset.tar.gz'
estimator.fit({'training': training_input})
```

## 📊 Monitoring and Evaluation

### Key Metrics to Track

1. **Training Metrics**:
   - Training loss
   - Box loss
   - Class loss
   - Object loss

2. **Validation Metrics**:
   - mAP@0.5
   - mAP@0.5:0.95
   - Precision
   - Recall

3. **Per-Class Performance**:
   - Class-specific mAP
   - Confusion matrix
   - F1 scores

### MLflow Tracking

```python
# Enhanced logging for custom datasets
def log_custom_metrics(results, class_names):
    """Log detailed metrics for custom dataset training"""
    
    # Log training curves
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        
        # Overall metrics
        mlflow.log_metric("train_loss", metrics.get('train/box_loss', 0))
        mlflow.log_metric("val_map50", metrics.get('metrics/mAP_0.5', 0))
        mlflow.log_metric("val_map50_95", metrics.get('metrics/mAP_0.5:0.95', 0))
        
        # Per-class metrics (if available)
        for i, class_name in enumerate(class_names):
            class_map = metrics.get(f'metrics/mAP_0.5({class_name})', None)
            if class_map is not None:
                mlflow.log_metric(f"map50_{class_name}", class_map)
    
    # Log training plots
    plots_dir = Path("runs/detect/train")
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(plot_file), "plots")
```

## 🎯 Optimization Tips

### 1. Hyperparameter Tuning

```python
# Hyperparameter search space
search_space = {
    'lr0': [0.001, 0.01, 0.1],
    'batch_size': [8, 16, 32],
    'image_size': [416, 640, 832],
    'epochs': [100, 150, 200]
}

# Example grid search
for lr in search_space['lr0']:
    for batch in search_space['batch_size']:
        with mlflow.start_run(run_name=f"lr{lr}_batch{batch}"):
            model = YOLO('yolo11s.pt')
            results = model.train(
                data='dataset.yaml',
                lr0=lr,
                batch=batch,
                epochs=50  # Shorter for search
            )
            
            # Log results
            mlflow.log_params({"lr0": lr, "batch_size": batch})
            # ... log metrics
```

### 2. Data Augmentation

```python
# Advanced augmentation settings
augmentation_config = {
    # Geometric
    'degrees': 10.0,      # Rotation
    'translate': 0.2,     # Translation
    'scale': 0.9,         # Scale
    'shear': 10.0,        # Shear
    'perspective': 0.001, # Perspective
    
    # Color
    'hsv_h': 0.02,        # Hue
    'hsv_s': 0.7,         # Saturation
    'hsv_v': 0.4,         # Value
    
    # Flips
    'fliplr': 0.5,        # Horizontal flip
    'flipud': 0.0,        # Vertical flip
    
    # Mixup augmentations
    'mosaic': 1.0,        # Mosaic probability
    'mixup': 0.15,        # Mixup probability
    'copy_paste': 0.3     # Copy-paste probability
}
```

### 3. Transfer Learning

```python
# Fine-tune from a pre-trained model
def fine_tune_model(pretrained_path, dataset_yaml, num_classes):
    """Fine-tune a pre-trained YOLO11 model"""
    
    # Load pre-trained model
    model = YOLO(pretrained_path)
    
    # Modify for new number of classes if needed
    if num_classes != model.model[-1].nc:
        print(f"Adapting model from {model.model[-1].nc} to {num_classes} classes")
    
    # Train with lower learning rate for fine-tuning
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        lr0=0.001,  # Lower learning rate
        warmup_epochs=5,
        freeze=10,  # Freeze first 10 layers
        patience=20
    )
    
    return model, results
```

## 🔍 Troubleshooting

### Common Issues

1. **Class Imbalance**:
   ```python
   # Use class weights
   class_weights = {0: 1.0, 1: 2.0, 2: 1.5}  # Adjust based on your data
   ```

2. **Memory Issues**:
   ```python
   # Reduce batch size and image size
   model.train(
       data='dataset.yaml',
       batch=8,    # Smaller batch
       imgsz=416,  # Smaller image size
       workers=2   # Fewer workers
   )
   ```

3. **Slow Convergence**:
   ```python
   # Adjust learning rate and warmup
   model.train(
       data='dataset.yaml',
       lr0=0.01,
       warmup_epochs=5,
       warmup_bias_lr=0.1
   )
   ```

## 📈 Best Practices

1. **Dataset Quality**:
   - Ensure consistent annotation quality
   - Use diverse training data
   - Include edge cases and difficult examples

2. **Training Strategy**:
   - Start with smaller models (yolo11n/s) for quick experimentation
   - Use transfer learning from COCO-pretrained models
   - Monitor validation metrics to avoid overfitting

3. **MLflow Organization**:
   - Use descriptive experiment names
   - Tag runs with dataset versions
   - Log hyperparameters and dataset statistics

4. **Resource Management**:
   - Use spot instances for cost savings
   - Scale instance types based on dataset size
   - Implement early stopping to save compute

## 🔗 Next Steps

- [Model Deployment Guide](model-deployment.md)
- [Advanced Configurations](advanced-config.md)
- [Performance Optimization](performance-optimization.md)

---

**Need help?** Create an issue in the repository or check the [FAQ](faq.md). m