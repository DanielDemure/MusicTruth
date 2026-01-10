#!/usr/bin/env python3
"""
Utility script to download and verify Deezer model.

Usage:
    python execution/download_deezer_model.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers.analysis.features.deepfake import DeezerDetector

def main():
    print("🔧 Deezer Model Download Utility")
    print("=" * 50)
    
    detector = DeezerDetector()
    
    # Check PyTorch
    if not detector.is_available():
        print("❌ PyTorch not installed!")
        print("   Install: pip install torch")
        return 1
    
    # Get model path
    model_path = detector._get_model_path()
    print(f"📂 Model path: {model_path}")
    
    # Check if exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists ({size_mb:.1f} MB)")
        return 0
    
    # Download
    print(f"📥 Downloading from: {detector.MODEL_URL}")
    success = detector._download_model()
    
    if success:
        print("✅ Download complete!")
        
        # Verify
        try:
            detector._load_model()
            if detector.model is not None:
                print("✅ Model loaded successfully!")
                return 0
            else:
                print("❌ Model failed to load")
                return 1
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return 1
    else:
        print("❌ Download failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
