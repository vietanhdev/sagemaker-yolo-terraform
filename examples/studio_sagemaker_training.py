#!/usr/bin/env python3
"""
SageMaker Studio YOLO Training Script
Reads configuration from files instead of hardcoding values.
Designed for Studio deployment mode.
"""

import sys
from utils import run_training


def main():
    """Main training function for Studio mode."""
    try:
        # Use the shared training function with studio configuration
        run_training(
            config_file="config_studio.yaml",
            deployment_mode="studio",
            script_name="studio"
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 