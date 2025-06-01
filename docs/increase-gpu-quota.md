# How to Increase AWS SageMaker GPU Instance Quotas

## Overview

AWS accounts have default quota limits for SageMaker training instances. GPU instances (like `ml.g4dn.xlarge`) often have a default limit of **0 instances**, which means you need to request an increase before using them for training.

> **🚨 IMPORTANT:**
> 
> **1. Request the RIGHT quota type:**
> - `use_spot_instances=False` → Request **"for training job usage"** (on-demand)
> - `use_spot_instances=True` → Request **"for spot training job usage"** (spot)
> - **Best practice:** Request BOTH for flexibility
> 
> **2. Wait for propagation:**
> - After approval email: **Wait 15-30 minutes** before testing
> - Some regions may take up to 1 hour
> 
> **3. Match script to quota:**
> - Your script's `use_spot_instances` setting must match the quota you received

## ⚠️ **Critical Information**

### **Spot vs On-Demand Quotas**
AWS has **separate quotas** for each instance type:
- **On-demand quota**: `ml.g4dn.xlarge for training job usage`
- **Spot quota**: `ml.g4dn.xlarge for spot training job usage`

**You need to request the appropriate quota based on your usage:**
- **If using `use_spot_instances=False`** → Request **on-demand quota**
- **If using `use_spot_instances=True`** → Request **spot quota**
- **For flexibility** → Request **both quotas**

### **Quota Propagation Time**
After quota approval:
- ⏰ **Wait 15-30 minutes** for changes to propagate
- 🔄 **AWS API caches** may take up to 1 hour to refresh
- 📧 **Email confirmation** doesn't mean immediate availability

## 🚨 Common Error

```
ResourceLimitExceeded: The account-level service limit 'ml.g4dn.xlarge for spot training job usage' is 0 Instances, with current utilization of 0 Instances and a request delta of 1 Instances.
```

This error means you need to request a quota increase.

## 🔧 Method 1: AWS Service Quotas Console (Recommended)

### Step 1: Access Service Quotas
1. **Log in to AWS Console**
2. **Search for "Service Quotas"** in the top search bar
3. **Or go directly to**: https://console.aws.amazon.com/servicequotas/

### Step 2: Navigate to SageMaker Quotas
1. **Click "AWS services"** in the left sidebar
2. **Search for "sagemaker"** and select **"Amazon SageMaker"**
3. **You'll see a list of all SageMaker quotas**

### Step 3: Find GPU Instance Quotas
Search for these common GPU instance quotas:
- `ml.g4dn.xlarge for training job usage` (on-demand)
- `ml.g4dn.xlarge for spot training job usage` (spot instances)
- `ml.g5.xlarge for training job usage` (newer GPU instances)
- `ml.p3.2xlarge for training job usage` (high-performance GPU)

### Step 4: Request Increase
1. **Click on the quota name** (e.g., "ml.g4dn.xlarge for training job usage")
2. **Click "Request quota increase"** button
3. **Fill out the request form**:

#### Form Fields:
- **New quota value**: `1` or `2` (start small)
- **Use case description**: See sample below
- **Contact method**: Email (recommended)

#### Sample Use Case Description:
```
I need to run YOLO machine learning training jobs using SageMaker for a computer vision project. 
I require ml.g4dn.xlarge instances for GPU acceleration to train object detection models efficiently. 

Project Details:
- Application: Custom YOLO model training for object detection
- Dataset: Proprietary image datasets for beverage container detection
- Expected Usage: 1-2 concurrent training jobs, 2-4 hours each
- Timeline: Development phase over next 3 months
- Business Purpose: Proof of concept for automated quality control system

This is for development and testing purposes. I'm willing to use alternative instance types 
(ml.g5.xlarge, ml.p3.2xlarge) if ml.g4dn.xlarge is not available.
```

### Step 5: Submit and Wait
- **Click "Request"**
- **Check your email** for confirmation
- **Most requests are processed within 24-48 hours**

## 🔧 Method 2: AWS CLI

### Check Current Quotas
```bash
# List all SageMaker quotas
aws service-quotas list-service-quotas \
    --service-code sagemaker \
    --region us-east-1 \
    --query 'Quotas[?contains(QuotaName, `g4dn.xlarge`)]' \
    --output table

# Check specific quota value
aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code L-8C7A6A1A \
    --region us-east-1
```

### Request Increase via CLI
```bash
# Request quota increase for ml.g4dn.xlarge on-demand
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-8C7A6A1A \
    --desired-value 2 \
    --region us-east-1

# Request quota increase for ml.g4dn.xlarge spot instances
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-D8E78DD9 \
    --desired-value 2 \
    --region us-east-1
```

### Check Request Status
```bash
# List all quota requests
aws service-quotas list-requested-service-quota-change-history \
    --service-code sagemaker \
    --region us-east-1

# Check specific request
aws service-quotas get-requested-service-quota-change \
    --request-id YOUR_REQUEST_ID \
    --region us-east-1
```

## 🔧 Method 3: AWS Support Center

### When to Use Support Center:
- Service Quotas console is not available
- Complex quota requirements
- Need assistance with planning

### Steps:
1. **Go to**: https://console.aws.amazon.com/support/
2. **Click "Create case"**
3. **Select "Service limit increase"**
4. **Fill out the form**:
   - **Service**: Amazon SageMaker
   - **Region**: Your target region (e.g., us-east-1)
   - **Resource Type**: Training instances
   - **Limit**: ml.g4dn.xlarge for training job usage
   - **New limit value**: 1 or 2

## 📊 Common SageMaker GPU Instance Quota Codes

| Instance Type | On-Demand Code | Spot Code | VRAM | Default Limit |
|---------------|----------------|-----------|------|---------------|
| `ml.g4dn.xlarge` | L-8C7A6A1A | L-D8E78DD9 | 16GB | 0 |
| `ml.g4dn.2xlarge` | L-C5B2E94B | L-7A8F9C2D | 32GB | 0 |
| `ml.g5.xlarge` | L-F7BC5BFC | L-7F2B9D7A | 24GB | 0 |
| `ml.g5.2xlarge` | L-8E3C4D5F | L-9A2B3C4E | 48GB | 0 |
| `ml.p3.2xlarge` | L-C5F122F4 | L-8B5F68A3 | 16GB | 0 |

## 🎯 **Common Scenarios & What to Request**

### Scenario 1: Cost-Optimized Training (Spot Instances)
```python
# Your SageMaker script uses:
use_spot_instances=True
```
**Request:** `ml.g4dn.xlarge for spot training job usage` (Code: L-D8E78DD9)

### Scenario 2: Reliable Training (On-Demand)
```python
# Your SageMaker script uses:
use_spot_instances=False
```
**Request:** `ml.g4dn.xlarge for training job usage` (Code: L-8C7A6A1A)

### Scenario 3: Maximum Flexibility (Recommended)
**Request BOTH quotas** so you can switch between spot and on-demand:
- `ml.g4dn.xlarge for training job usage` (on-demand)
- `ml.g4dn.xlarge for spot training job usage` (spot)

### Scenario 4: Still Getting Errors After Approval
**Common Issue:** Script configuration doesn't match requested quota

**Check your script:**
```python
# If you requested on-demand quota, make sure you have:
use_spot_instances=False

# If you requested spot quota, make sure you have:
use_spot_instances=True
```

## ⏰ **Quota Propagation & Troubleshooting**

### After Getting Approval Email:

#### Step 1: Wait for Propagation (15-30 minutes)
```bash
# Don't run your training job immediately
# Wait at least 15-30 minutes after approval email
```

#### Step 2: Verify Quota Changes
```bash
# Check if quota actually increased
aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code L-8C7A6A1A \
    --region us-east-1

# Expected output should show new quota value > 0
```

#### Step 3: Clear Local AWS CLI Cache
```bash
# Clear any cached credentials/settings
aws configure list
export AWS_CLI_CACHE_DIR=""
```

#### Step 4: Match Script to Quota Type
```python
# If you got ON-DEMAND quota approval:
use_spot_instances=False

# If you got SPOT quota approval:
use_spot_instances=True
```

#### Step 5: Try Again
```bash
python examples/custom_sagemaker_training.py
```

### Still Getting Errors?

#### Wait Longer (Up to 1 Hour)
Some AWS regions take longer to propagate quota changes.

#### Check Different Region
Your quota might be region-specific:
```bash
# Verify you're requesting quota in the same region as your training
aws configure get region
```

#### Try Alternative Instance Type
```python
# Change your script to use a different GPU instance:
instance_type="ml.g5.xlarge"    # Instead of ml.g4dn.xlarge
```

## 🎯 Best Practices for Quota Requests

### 1. Start Small
- **Request 1-2 instances** initially
- **Scale up later** as needed
- Shows responsible usage

### 2. Be Specific About Usage
Include in your request:
- **Project description**
- **Expected training frequency**
- **Duration per training job**
- **Business justification**

### 3. Have Alternatives Ready
Mention flexibility:
- "Willing to use ml.g5.xlarge if ml.g4dn.xlarge not available"
- "Can use different regions if needed"
- "Prefer on-demand over spot if quotas allow"

### 4. Choose the Right Region
Some regions have better GPU availability:
```bash
# Check instance availability by region
aws ec2 describe-instance-type-offerings \
    --location-type region \
    --filters Name=instance-type,Values=g4dn.xlarge \
    --query 'InstanceTypeOfferings[].Location' \
    --output table
```

Popular regions for ML workloads:
- `us-east-1` (N. Virginia) - Usually best availability
- `us-west-2` (Oregon) - Good for ML services
- `eu-west-1` (Ireland) - Good for European users

## ⏱️ Expected Timeline

| Request Type | Response Time | Auto-Approval |
|-------------|---------------|---------------|
| **Standard GPU instances** | 1-24 hours | Often ✅ |
| **High-end instances (p3, p4)** | 1-3 days | Sometimes |
| **Large quantities (>5)** | 2-5 days | Rarely |
| **Complex business cases** | 3-7 days | No |

## 🔍 Monitoring Your Requests

### Via AWS Console:
1. **Service Quotas** → **Dashboard**
2. **"Quota request history"** tab
3. **Check status**: Pending, Approved, Denied

### Via CLI:
```bash
# Check all recent requests
aws service-quotas list-requested-service-quota-change-history \
    --service-code sagemaker \
    --region us-east-1 \
    --query 'RequestedQuotas[?Status==`PENDING`]'
```

## 🚀 Quick Action Checklist

### For YOLO Training Project:

- [ ] **Access Service Quotas Console**
- [ ] **Search for "Amazon SageMaker"**
- [ ] **Find "ml.g4dn.xlarge for training job usage"** (on-demand)
- [ ] **Request increase to 1-2 instances**
- [ ] **ALSO find "ml.g4dn.xlarge for spot training job usage"** (spot)
- [ ] **Request increase to 1-2 instances** (for flexibility)
- [ ] **Use the sample justification above**
- [ ] **Select email notifications**
- [ ] **Submit both requests**
- [ ] **Check email for updates**
- [ ] **Wait 15-30 minutes after approval before testing**

### After Quota Approval:
- [ ] **Wait 15-30 minutes** for propagation
- [ ] **Verify quota with AWS CLI** (see commands above)
- [ ] **Match your script settings** to approved quota type
- [ ] **Test your training job**

### Alternative if GPU Denied:
- [ ] **Try ml.g5.xlarge** (newer generation)
- [ ] **Try different region** (us-west-2, eu-west-1)
- [ ] **Use CPU instances** with smaller models for testing

## 💰 Cost Considerations

| Instance | On-Demand $/hr | Spot $/hr | Use Case |
|----------|----------------|-----------|----------|
| `ml.g4dn.xlarge` | ~$0.736 | ~$0.22-0.37 | Standard YOLO training |
| `ml.g5.xlarge` | ~$1.408 | ~$0.42-0.70 | Faster training, newer GPU |
| `ml.m5.large` | ~$0.096 | ~$0.029 | CPU fallback option |

**Recommendation**: Start with **on-demand** to avoid spot quota issues, then switch to spot once quotas are approved.

## 🔧 Troubleshooting

### Common Issues:

#### 1. "No quota increase option available"
- **Solution**: Use AWS Support Center instead
- **Reason**: Some accounts need manual review

#### 2. "Request denied"
- **Solution**: Provide more detailed business justification
- **Try**: Different instance type or smaller quantity

#### 3. "Quota approved but still getting errors"
- **Wait**: Propagation can take 15-30 minutes
- **Check**: Correct region and instance type
- **Verify**: On-demand vs spot quota

#### 4. "Cannot find quota in Service Quotas"
- **Search**: Use different keywords ("g4dn", "training", "gpu")
- **Region**: Make sure you're in the correct region
- **Alternative**: Use AWS Support Center

## 📞 Getting Help

### AWS Support Resources:
- **Service Quotas Documentation**: https://docs.aws.amazon.com/servicequotas/
- **SageMaker Limits**: https://docs.aws.amazon.com/sagemaker/latest/dg/limits.html
- **Support Center**: https://console.aws.amazon.com/support/

### Community Resources:
- **AWS Forums**: https://forums.aws.amazon.com/
- **Stack Overflow**: Tag questions with `amazon-sagemaker`
- **AWS re:Post**: https://repost.aws/

## 📝 Request Template

Copy this template for your quota request:

```
Subject: SageMaker GPU Training Instance Quota Increase

Service: Amazon SageMaker
Region: us-east-1
Current Limit: 0
Requested Limit: 2
Instance Type: ml.g4dn.xlarge for training job usage

Business Justification:
I'm developing a computer vision application using YOLO models for object detection 
as part of a machine learning research project. The application will detect and 
classify objects in images for automated quality control.

Technical Requirements:
- Framework: PyTorch with Ultralytics YOLO
- Model Types: YOLOv8, YOLO11 variants
- Training Data: Custom labeled datasets (~10K images)
- GPU Memory: 16GB+ required for batch training
- Expected Usage: 2-3 training jobs per week, 2-4 hours each

This is for development and prototyping purposes. I'm willing to use alternative 
instance types (ml.g5.xlarge) if ml.g4dn.xlarge is not available.

Thank you for your consideration.
```

---

**Next Steps**: After quota approval, you can use GPU instances in your SageMaker training jobs! 🚀 