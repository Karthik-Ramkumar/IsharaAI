"""
Script: train.py
Description: Train the ISL gesture classifier
Author: Hackathon Team
Date: 2026

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 100 --batch-size 64
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from models.train_model import full_training_pipeline, TF_AVAILABLE, SKLEARN_AVAILABLE


def main():
    parser = argparse.ArgumentParser(description='Train ISL gesture classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--data-dir', type=str, default=None, 
                       help='Preprocessed data directory')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Model save directory')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ISL Gesture Classifier Training")
    print("=" * 50)
    
    # Check dependencies
    if not TF_AVAILABLE:
        print("\n❌ Error: TensorFlow not installed")
        print("Install with: pip install tensorflow")
        sys.exit(1)
    
    if not SKLEARN_AVAILABLE:
        print("\n❌ Warning: scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        print("Continuing without advanced metrics...")
    
    # Set paths
    data_dir = args.data_dir or config.LANDMARKS_DIR
    model_dir = args.model_dir or config.MODELS_DIR
    
    print(f"\nData directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Check if preprocessed data exists
    if not os.path.exists(os.path.join(data_dir, 'X.npy')):
        print(f"\n❌ Error: Preprocessed data not found in {data_dir}")
        print("\nRun preprocessing first:")
        print("   python scripts/preprocess_dataset.py")
        sys.exit(1)
    
    # Run training
    print("\n" + "=" * 50)
    success = full_training_pipeline(
        data_dir=data_dir,
        model_dir=model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if success:
        print("\n🎉 Training successful!")
        print("\n🚀 Next steps:")
        print("   1. Test with: python scripts/test_camera.py")
        print("   2. Run app: python app.py")
    else:
        print("\n❌ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
