#!/usr/bin/env python3
"""
SageMaker Custom YOLO Training Script
Reads configuration from files instead of hardcoding values.
Designed for custom deployment mode (EC2-based MLflow with RDS).
"""

import sys
from utils import run_training


def main():
    """Main training function for custom mode."""
    try:
        # Use the shared training function with custom configuration
        estimator = run_training(
            config_file="config_custom.yaml",
            deployment_mode="custom",
            script_name="custom"
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()