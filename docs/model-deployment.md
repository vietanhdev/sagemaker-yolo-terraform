# Model Deployment Guide

This guide covers deploying your trained YOLO11 models for inference using various deployment strategies, from cloud endpoints to edge devices.

## 🎯 Deployment Options Overview

| Deployment Type | Use Case | Latency | Cost | Scalability |
|-----------------|----------|---------|------|-------------|
| **SageMaker Real-time** | Production APIs | Low | Medium-High | Auto-scaling |
| **SageMaker Serverless** | Intermittent requests | Medium | Low | Auto-scaling |
| **SageMaker Batch** | Bulk processing | High | Low | Batch jobs |
| **Docker Container** | Custom environments | Low | Variable | Manual |
| **Edge Deployment** | Mobile/IoT | Very Low | Very Low | Fixed |

## 🚀 SageMaker Real-time Endpoints

### Step 1: Prepare Model for Deployment

```python
#!/usr/bin/env python3
"""Prepare YOLO11 model for SageMaker deployment"""

import torch
import json
from pathlib import Path
from ultralytics import YOLO
import mlflow

def export_model_for_sagemaker(model_path, output_dir):
    """Export YOLO11 model for SageMaker deployment"""
    
    # Load trained model
    model = YOLO(model_path)
    
    # Export to TorchScript for better performance
    model.export(
        format='torchscript',
        optimize=True,
        half=False,  # Set True for FP16 optimization
        device='cpu'
    )
    
    # Create model artifacts directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Copy exported model
    torchscript_path = Path(model_path).with_suffix('.torchscript')
    if torchscript_path.exists():
        import shutil
        shutil.copy2(torchscript_path, output_path / 'model.pt')
    
    # Create model configuration
    model_config = {
        "model_type": "yolo11",
        "input_shape": [640, 640, 3],
        "num_classes": model.model[-1].nc,
        "class_names": model.names,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45
    }
    
    with open(output_path / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model exported to: {output_path}")
    return output_path

# Usage example
model_artifacts = export_model_for_sagemaker(
    model_path="runs/detect/train/weights/best.pt",
    output_dir="model_artifacts"
)
```

### Step 2: Create Inference Script

```python
# inference.py
"""SageMaker inference script for YOLO11"""

import json
import logging
import os
import torch
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model for inference"""
    logger.info("Loading YOLO11 model...")
    
    # Load TorchScript model
    model_path = os.path.join(model_dir, 'model.pt')
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    # Load configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'config': config
    }

def input_fn(request_body, request_content_type='application/json'):
    """Parse input data for inference"""
    logger.info(f"Input content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle base64 encoded image
        if 'image' in input_data:
            image_data = base64.b64decode(input_data['image'])
            image = Image.open(io.BytesIO(image_data))
            return {
                'image': image,
                'confidence': input_data.get('confidence', 0.25),
                'iou_threshold': input_data.get('iou_threshold', 0.45)
            }
    
    elif request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        image = Image.open(io.BytesIO(request_body))
        return {
            'image': image,
            'confidence': 0.25,
            'iou_threshold': 0.45
        }
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_data):
    """Run inference"""
    model = model_data['model']
    config = model_data['config']
    
    # Preprocess image
    image = input_data['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to model input size
    img_size = config['input_shape'][:2]  # [height, width]
    image_resized = image.resize((img_size[1], img_size[0]))
    
    # Convert to tensor
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Post-process results
    results = post_process_predictions(
        predictions, 
        config,
        input_data.get('confidence', config['confidence_threshold']),
        input_data.get('iou_threshold', config['iou_threshold'])
    )
    
    return results

def post_process_predictions(predictions, config, conf_threshold, iou_threshold):
    """Post-process YOLO predictions"""
    # This is a simplified post-processing
    # In practice, you'd use the full YOLO post-processing pipeline
    
    results = []
    class_names = config['class_names']
    
    # Extract predictions (simplified)
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    
    # Convert to detections format
    # Format: [x1, y1, x2, y2, confidence, class_id]
    
    # Apply confidence threshold
    confident_predictions = predictions[predictions[:, 4] > conf_threshold]
    
    # Apply NMS (simplified version)
    # In practice, use torchvision.ops.nms or ultralytics NMS
    
    for pred in confident_predictions[:50]:  # Limit to top 50
        x1, y1, x2, y2, conf, class_id = pred[:6]
        results.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(conf),
            'class_id': int(class_id),
            'class_name': class_names.get(str(int(class_id)), 'unknown')
        })
    
    return {
        'predictions': results,
        'num_detections': len(results)
    }

def output_fn(prediction, accept='application/json'):
    """Format the prediction output"""
    logger.info(f"Output accept type: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction), accept
    
    raise ValueError(f"Unsupported accept type: {accept}")
```

### Step 3: Deploy to SageMaker

```python
#!/usr/bin/env python3
"""Deploy YOLO11 model to SageMaker endpoint"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import tarfile
import os

def create_model_tarball(model_dir, output_path):
    """Create model.tar.gz for SageMaker"""
    with tarfile.open(output_path, 'w:gz') as tar:
        tar.add(model_dir, arcname='.')
    print(f"Model tarball created: {output_path}")

def deploy_yolo_model(
    model_artifacts_path,
    model_name,
    endpoint_name,
    instance_type='ml.m5.xlarge',
    instance_count=1
):
    """Deploy YOLO11 model to SageMaker endpoint"""
    
    # Create SageMaker session
    session = sagemaker.Session()
    role = get_execution_role()
    
    # Create model tarball
    tarball_path = 'model.tar.gz'
    create_model_tarball(model_artifacts_path, tarball_path)
    
    # Upload to S3
    bucket = session.default_bucket()
    model_s3_path = session.upload_data(
        path=tarball_path,
        bucket=bucket,
        key_prefix=f'yolo11-models/{model_name}'
    )
    
    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_s3_path,
        role=role,
        entry_point='inference.py',
        source_dir='deployment_scripts',  # Directory containing inference.py
        framework_version='2.0.0',
        py_version='py310',
        model_server_timeout=60,
        model_server_workers=1,
        env={
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '60',
            'SAGEMAKER_MODEL_SERVER_WORKERS': '1'
        }
    )
    
    # Deploy model
    predictor = pytorch_model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        wait=True
    )
    
    print(f"Model deployed to endpoint: {endpoint_name}")
    return predictor

# Usage example
predictor = deploy_yolo_model(
    model_artifacts_path='model_artifacts',
    model_name='yolo11s-custom',
    endpoint_name='yolo11-endpoint',
    instance_type='ml.m5.large'
)
```

### Step 4: Test the Endpoint

```python
#!/usr/bin/env python3
"""Test YOLO11 SageMaker endpoint"""

import boto3
import json
import base64
from PIL import Image
import io

def test_yolo_endpoint(endpoint_name, image_path):
    """Test YOLO11 endpoint with an image"""
    
    # Initialize SageMaker runtime
    runtime = boto3.client('sagemaker-runtime')
    
    # Prepare image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Encode image as base64
    image_b64 = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare request
    payload = {
        'image': image_b64,
        'confidence': 0.3,
        'iou_threshold': 0.5
    }
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    
    print(f"Number of detections: {result['num_detections']}")
    for i, detection in enumerate(result['predictions'][:5]):  # Show top 5
        print(f"Detection {i+1}:")
        print(f"  Class: {detection['class_name']}")
        print(f"  Confidence: {detection['confidence']:.3f}")
        print(f"  BBox: {detection['bbox']}")
    
    return result

# Test the endpoint
results = test_yolo_endpoint(
    endpoint_name='yolo11-endpoint',
    image_path='test_image.jpg'
)
```

## 🔄 SageMaker Serverless Inference

For intermittent or unpredictable traffic:

```python
from sagemaker.serverless import ServerlessInferenceConfig

# Deploy as serverless endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=6144,  # 6GB memory
    max_concurrency=10,      # Max concurrent invocations
    provisioned_concurrency=1  # Keep 1 instance warm
)

predictor = pytorch_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='yolo11-serverless'
)
```

## 📦 Docker Container Deployment

### Dockerfile for YOLO11

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
```

### Flask API for YOLO11

```python
# app.py
"""Flask API for YOLO11 inference"""

from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model globally
MODEL = None

def load_model():
    """Load YOLO11 model"""
    global MODEL
    if MODEL is None:
        MODEL = YOLO('best.pt')  # Your trained model
        MODEL.to('cuda' if torch.cuda.is_available() else 'cpu')
    return MODEL

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': MODEL is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Load model
        model = load_model()
        
        # Get image from request
        if 'image' in request.files:
            # File upload
            image_file = request.files['image']
            image = Image.open(image_file.stream)
        elif request.json and 'image' in request.json:
            # Base64 encoded image
            image_data = base64.b64decode(request.json['image'])
            image = Image.open(io.BytesIO(image_data))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get parameters
        confidence = float(request.form.get('confidence', 0.25))
        iou_threshold = float(request.form.get('iou_threshold', 0.45))
        
        # Run inference
        results = model(
            image,
            conf=confidence,
            iou=iou_threshold,
            verbose=False
        )
        
        # Format results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': model.names[int(box.cls[0])]
                    }
                    detections.append(detection)
        
        return jsonify({
            'predictions': detections,
            'num_detections': len(detections),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

### Build and Run Docker Container

```bash
# Build the image
docker build -t yolo11-api:latest .

# Run the container
docker run -p 8080:8080 --gpus all yolo11-api:latest

# Test the API
curl -X POST \
  -F "image=@test_image.jpg" \
  -F "confidence=0.3" \
  http://localhost:8080/predict
```

## 📱 Edge Deployment

### Export for Mobile Devices

```python
#!/usr/bin/env python3
"""Export YOLO11 for mobile/edge deployment"""

from ultralytics import YOLO

def export_for_mobile(model_path, export_formats=['onnx', 'tflite', 'coreml']):
    """Export model for various edge deployment formats"""
    
    model = YOLO(model_path)
    
    exported_models = {}
    
    for format_type in export_formats:
        print(f"Exporting to {format_type}...")
        
        if format_type == 'onnx':
            # ONNX for general deployment
            exported_path = model.export(
                format='onnx',
                optimize=True,
                half=False,
                simplify=True,
                opset=11
            )
            
        elif format_type == 'tflite':
            # TensorFlow Lite for Android/edge
            exported_path = model.export(
                format='tflite',
                int8=True,  # INT8 quantization
                optimize=True
            )
            
        elif format_type == 'coreml':
            # Core ML for iOS
            exported_path = model.export(
                format='coreml',
                half=True,
                int8=False
            )
            
        exported_models[format_type] = exported_path
        print(f"✓ {format_type}: {exported_path}")
    
    return exported_models

# Export model
exported = export_for_mobile('best.pt')
```

### ONNX Runtime Inference

```python
#!/usr/bin/env python3
"""ONNX Runtime inference for edge deployment"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

class YOLOONNXInference:
    def __init__(self, model_path, providers=['CPUExecutionProvider']):
        """Initialize ONNX model"""
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize image
        h, w = self.input_shape[2:4]
        image_resized = cv2.resize(image, (w, h))
        
        # Normalize
        image_norm = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose
        image_batch = np.expand_dims(image_norm.transpose(2, 0, 1), axis=0)
        
        return image_batch
    
    def predict(self, image):
        """Run inference"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        return outputs[0]  # Return first output
    
    def postprocess(self, predictions, conf_threshold=0.25, iou_threshold=0.45):
        """Post-process predictions"""
        # Implement NMS and filtering
        # This is a simplified version
        
        detections = []
        for pred in predictions[0]:  # Assuming batch size 1
            confidence = pred[4]
            if confidence > conf_threshold:
                detection = {
                    'bbox': pred[:4].tolist(),
                    'confidence': float(confidence),
                    'class_id': int(pred[5:].argmax())
                }
                detections.append(detection)
        
        return detections

# Usage
model = YOLOONNXInference('best.onnx')
image = cv2.imread('test_image.jpg')
predictions = model.predict(image)
detections = model.postprocess(predictions)
```

## 🔧 Performance Optimization

### Model Optimization Techniques

```python
#!/usr/bin/env python3
"""Model optimization for deployment"""

from ultralytics import YOLO
import torch

def optimize_model_for_deployment(model_path, optimization_level='standard'):
    """Optimize YOLO11 model for deployment"""
    
    model = YOLO(model_path)
    
    if optimization_level == 'standard':
        # TorchScript compilation
        model.export(
            format='torchscript',
            optimize=True,
            half=False
        )
        
    elif optimization_level == 'aggressive':
        # Half precision + optimization
        model.export(
            format='torchscript',
            optimize=True,
            half=True,
            simplify=True
        )
        
    elif optimization_level == 'quantized':
        # INT8 quantization
        model.export(
            format='onnx',
            optimize=True,
            half=False,
            int8=True
        )
    
    print(f"Model optimized with {optimization_level} settings")

# Performance benchmarking
def benchmark_model(model_path, input_shape=(1, 3, 640, 640), num_runs=100):
    """Benchmark model inference speed"""
    import time
    
    model = torch.jit.load(model_path)
    model.eval()
    
    # Warm-up
    dummy_input = torch.randn(input_shape)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    return avg_time, fps
```

## 📊 Monitoring and Scaling

### CloudWatch Monitoring

```python
#!/usr/bin/env python3
"""Set up monitoring for deployed models"""

import boto3

def setup_endpoint_monitoring(endpoint_name):
    """Set up CloudWatch monitoring for SageMaker endpoint"""
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Create custom metrics dashboard
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/SageMaker", "InvocationsPerInstance", "EndpointName", endpoint_name],
                        ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_name],
                        ["AWS/SageMaker", "OverheadLatency", "EndpointName", endpoint_name]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": f"YOLO11 Endpoint Metrics - {endpoint_name}"
                }
            }
        ]
    }
    
    # Create dashboard
    cloudwatch.put_dashboard(
        DashboardName=f'YOLO11-{endpoint_name}',
        DashboardBody=json.dumps(dashboard_body)
    )
    
    # Set up alarms
    cloudwatch.put_metric_alarm(
        AlarmName=f'{endpoint_name}-HighLatency',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=2,
        MetricName='ModelLatency',
        Namespace='AWS/SageMaker',
        Period=300,
        Statistic='Average',
        Threshold=5000.0,  # 5 seconds
        ActionsEnabled=True,
        AlarmDescription='Alert when model latency is high',
        Dimensions=[
            {
                'Name': 'EndpointName',
                'Value': endpoint_name
            }
        ]
    )
    
    print(f"Monitoring set up for endpoint: {endpoint_name}")
```

### Auto-scaling Configuration

```python
#!/usr/bin/env python3
"""Configure auto-scaling for SageMaker endpoint"""

import boto3

def setup_autoscaling(endpoint_name, variant_name='AllTraffic'):
    """Set up auto-scaling for SageMaker endpoint"""
    
    autoscaling_client = boto3.client('application-autoscaling')
    
    # Register scalable target
    autoscaling_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=1,
        MaxCapacity=5
    )
    
    # Create scaling policy
    autoscaling_client.put_scaling_policy(
        PolicyName=f'{endpoint_name}-scaling-policy',
        ServiceNamespace='sagemaker',
        ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 70.0,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleOutCooldown': 300,
            'ScaleInCooldown': 300
        }
    )
    
    print(f"Auto-scaling configured for endpoint: {endpoint_name}")
```

## 🧪 A/B Testing

```python
#!/usr/bin/env python3
"""Set up A/B testing for model variants"""

from sagemaker.model import Model
from sagemaker.predictor import Predictor

def create_multi_variant_endpoint(
    models_config,
    endpoint_name,
    traffic_distribution=None
):
    """Create endpoint with multiple model variants for A/B testing"""
    
    if traffic_distribution is None:
        # Equal traffic distribution
        num_variants = len(models_config)
        traffic_per_variant = 100 // num_variants
        traffic_distribution = [traffic_per_variant] * num_variants
        traffic_distribution[0] += 100 - sum(traffic_distribution)  # Handle remainder
    
    # Create endpoint configuration with variants
    endpoint_config = []
    
    for i, (variant_name, model_config) in enumerate(models_config.items()):
        variant_config = {
            'VariantName': variant_name,
            'ModelName': model_config['model_name'],
            'InitialInstanceCount': 1,
            'InstanceType': model_config.get('instance_type', 'ml.m5.large'),
            'InitialVariantWeight': traffic_distribution[i]
        }
        endpoint_config.append(variant_config)
    
    # Deploy endpoint with variants
    sagemaker_session = sagemaker.Session()
    sagemaker_session.create_endpoint_config(
        name=f'{endpoint_name}-config',
        model_name=models_config[list(models_config.keys())[0]]['model_name'],
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )
    
    print(f"Multi-variant endpoint created: {endpoint_name}")
    return endpoint_name

# Example usage
models_config = {
    'yolo11s-v1': {
        'model_name': 'yolo11s-model-v1',
        'instance_type': 'ml.m5.large'
    },
    'yolo11s-v2': {
        'model_name': 'yolo11s-model-v2', 
        'instance_type': 'ml.m5.large'
    }
}

endpoint_name = create_multi_variant_endpoint(
    models_config,
    'yolo11-ab-test',
    traffic_distribution=[70, 30]  # 70% to v1, 30% to v2
)
```

## 💰 Cost Optimization

### Spot Instance Usage

```python
# Use spot instances for batch inference
from sagemaker.transformer import Transformer

transformer = Transformer(
    model_name='yolo11-model',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    use_spot_instances=True,
    max_wait=3600,  # Wait up to 1 hour for spot capacity
    output_path='s3://bucket/batch-inference-output/'
)

# Run batch transform
transformer.transform(
    data='s3://bucket/input-images/',
    content_type='image/jpeg',
    split_type='None'
)
```

### Serverless for Variable Workloads

```python
# Use serverless inference for cost optimization
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3008,
    max_concurrency=5,
    provisioned_concurrency=0  # Only pay when used
)

predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='yolo11-cost-optimized'
)
```

## 🔗 Next Steps

- [Advanced Configurations](advanced-config.md)
- [Performance Optimization](performance-optimization.md)
- [Custom Dataset Training](custom-dataset.md)

---

**Need help with deployment?** Check the [FAQ](faq.md) or create an issue! 