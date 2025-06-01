# Configuration-Driven SageMaker YOLO Training Scripts

This directory contains configuration-driven training scripts that work with both Studio and Custom deployment modes, eliminating hardcoded values and making training easily configurable.

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `studio_sagemaker_training.py` | Studio mode training script |
| `custom_sagemaker_training.py` | Custom mode training script |
| `utils.py` | Shared utilities for both scripts |
| `config_studio.yaml` | Configuration for Studio mode |
| `config_custom.yaml` | Configuration for Custom mode |
| `requirements.txt` | Python dependencies |

## ✨ Key Features

- **🔧 Configuration-driven**: No hardcoded values in Python scripts
- **🤖 Auto-detection**: Reads deployment settings from `deployment_info.txt`
- **🔄 Mode-specific**: Automatically uses correct execution roles and MLflow URLs
- **📝 Flexible overrides**: Override any parameter via config files
- **✅ Error handling**: Validates configuration and deployment modes
- **🕐 Timestamped runs**: Generates unique MLflow run names
- **🎯 Shared code**: Common utilities prevent code duplication

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd examples
pip install -r requirements.txt
```

### 2. Studio Mode Training
```bash
# Edit config_studio.yaml if needed (optional)
python studio_sagemaker_training.py
```

### 3. Custom Mode Training
```bash
# Edit config_custom.yaml if needed (optional)
python custom_sagemaker_training.py
```

## ⚙️ Configuration

### Studio Mode (`config_studio.yaml`)
Used with `studio_sagemaker_training.py`:

```yaml
# Model Configuration
model:
  size: "yolo11s"        # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  epochs: 25
  batch_size: 16
  image_size: 640

# Infrastructure
sagemaker:
  instance_type: "ml.g4dn.xlarge"
  use_spot_instances: true
  max_run_hours: 4

# Data
data:
  dataset_name: "beverages"
  data_yaml_filename: "data.yaml"

# Experiment
experiment:
  name: "studio-yolo-training"
  run_name_prefix: "yolo"
```

### Custom Mode (`config_custom.yaml`)
Used with `custom_sagemaker_training.py`:

```yaml
# Same structure as studio config, but with:
experiment:
  name: "custom-yolo-training"  # Different experiment name
```

### Configuration Override Options
Both config files support overrides:

```yaml
overrides:
  s3_bucket: ""          # Override S3 bucket
  execution_role: ""     # Override IAM role
  mlflow_uri: ""         # Override MLflow URI
```

## 🔧 How It Works

### 1. **Auto-Detection Process**
```
1. Load config_studio.yaml or config_custom.yaml
2. Read ../deployment_info.txt for deployment settings
3. Validate deployment mode matches script
4. Build training parameters from both sources
5. Create SageMaker estimator
6. Launch training job
```

### 2. **Parameter Sources Priority**
```
1. Config file overrides (highest priority)
2. deployment_info.txt values
3. Default values (lowest priority)
```

### 3. **Deployment Info Mapping**

| Mode | S3 Bucket | Execution Role | MLflow URI |
|------|-----------|----------------|------------|
| **Studio** | `S3_BUCKET` | `STUDIO_EXECUTION_ROLE` | `STUDIO_MLFLOW_URL` |
| **Custom** | `S3_BUCKET` | `CUSTOM_SAGEMAKER_EXECUTION_ROLE` | `MLFLOW_UI_URL` |

## 📊 Example Output

### Studio Mode
```
📋 Loading configuration...
🚀 Starting SageMaker Studio training job...
   Using role: arn:aws:iam::123456789012:role/yolo-mlflow-studio-execution-role-xyz
   Instance type: ml.g4dn.xlarge
   Model: yolo11s
   Epochs: 25
   Batch size: 16
   Image size: 640
   MLflow URI: https://t-xyz.us-east-1.experiments.sagemaker.aws
   Experiment: studio-yolo-training
   Run name: yolo_20250601_142530
   Training data: s3://yolo-mlflow-artifacts-xyz/datasets/beverages/
   Spot instances: True
   Max runtime: 4 hours

🎯 Launching studio training job...
✅ Training job submitted successfully!
```

### Custom Mode
```
📋 Loading configuration...
🚀 Starting SageMaker Custom training job...
   Using role: arn:aws:iam::123456789012:role/yolo-mlflow-sagemaker-execution-role-xyz
   Instance type: ml.g4dn.xlarge
   Model: yolo11s
   Epochs: 25
   Batch size: 16
   Image size: 640
   MLflow URI: http://107.21.1.121:5000
   Experiment: custom-yolo-training
   Run name: yolo_20250601_142530
   Training data: s3://yolo-mlflow-artifacts-xyz/datasets/beverages/
   Spot instances: True
   Max runtime: 4 hours

🎯 Launching custom training job...
✅ Training job submitted successfully!
```

## 🔧 Customization Examples

### Change Model Size
```yaml
# config_studio.yaml or config_custom.yaml
model:
  size: "yolo11m"  # Larger model for better accuracy
```

### Use CPU Instance
```yaml
sagemaker:
  instance_type: "ml.m5.large"      # CPU instance
  use_spot_instances: false         # More reliable for CPU
```

### Override S3 Bucket
```yaml
overrides:
  s3_bucket: "my-custom-bucket"
```

### Longer Training
```yaml
model:
  epochs: 100              # More epochs
sagemaker:
  max_run_hours: 8         # Longer timeout
```

## 🛠️ Troubleshooting

### Error: "This script is for [mode] mode only"
**Solution**: Check `DEPLOYMENT_MODE` in `deployment_info.txt`
```bash
grep DEPLOYMENT_MODE ../deployment_info.txt
```

### Error: "Missing required deployment info"
**Solution**: Verify deployment_info.txt contains all required fields
```bash
# For Studio mode:
grep -E "S3_BUCKET|STUDIO_EXECUTION_ROLE|STUDIO_MLFLOW_URL" ../deployment_info.txt

# For Custom mode:
grep -E "S3_BUCKET|CUSTOM_SAGEMAKER_EXECUTION_ROLE|MLFLOW_UI_URL" ../deployment_info.txt
```

### Error: "Configuration file not found"
**Solution**: Ensure config files exist
```bash
ls -la config*.yaml
```

### Training Job Fails to Start
**Possible causes**:
- S3 bucket permissions
- SageMaker quotas
- Invalid instance type
- Missing dataset

**Debug steps**:
```bash
# Check S3 bucket access
aws s3 ls s3://your-bucket-name/

# Check SageMaker quotas
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-1194F91F  # ml.g4dn.xlarge quota
```

## 🔄 Migration from Legacy Scripts

### Old Hardcoded Approach
```python
role_arn = "arn:aws:iam::123456789012:role/hardcoded-role"
estimator = PyTorch(
    role=role_arn,
    instance_type="ml.g4dn.xlarge",
    # ... more hardcoded values
)
```

### New Configuration Approach
```python
from utils import run_training

estimator = run_training(
    config_file="config_studio.yaml",
    deployment_mode="studio"
)
```

## 📚 Advanced Usage

### Custom Configuration File
```python
# Use a custom config file
from utils import run_training

estimator = run_training(
    config_file="my_custom_config.yaml",
    deployment_mode="studio"
)
```

### Direct Utility Usage
```python
from utils import load_config, load_deployment_info, create_estimator

config = load_config("config_studio.yaml")
deployment_info = load_deployment_info("../deployment_info.txt")
# ... custom logic ...
estimator = create_estimator(config, parameters)
```

---

## 🎯 Next Steps

1. **Upload datasets** to your S3 bucket under `datasets/` folder
2. **Customize configs** for your specific training needs
3. **Monitor training** in MLflow UI (Studio or Custom)
4. **Scale training** with different instance types
5. **Experiment** with different YOLO model sizes

For more information, see the main [README.md](../README.md) in the project root. 