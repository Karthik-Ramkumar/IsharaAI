"""
Script: preprocess_dataset.py
Description: Convert raw ISL images to landmark format
Author: Hackathon Team
Date: 2026

Usage:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --max-per-class 100
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from src.preprocessing.dataset_converter import DatasetConverter


def main():
    parser = argparse.ArgumentParser(description='Preprocess ISL dataset')
    parser.add_argument('--max-per-class', type=int, default=None,
                       help='Maximum samples per class (None for all)')
    parser.add_argument('--raw-dir', type=str, default=None,
                       help='Raw data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    args = parser.parse_args()
    
    # Set paths
    raw_dir = args.raw_dir or config.RAW_DATA_DIR
    output_dir = args.output_dir or config.LANDMARKS_DIR
    
    print("=" * 50)
    print("ISL Dataset Preprocessing")
    print("=" * 50)
    print(f"\nRaw data: {raw_dir}")
    print(f"Output: {output_dir}")
    
    # Check if raw data exists
    if not os.path.exists(raw_dir):
        print(f"\n❌ Error: Raw data directory not found: {raw_dir}")
        print("\nPlease download the ISL dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl")
        sys.exit(1)
    
    # Create converter
    converter = DatasetConverter(raw_dir, output_dir)
    
    if not converter.is_available:
        print("\n❌ Error: MediaPipe not available")
        print("Install with: pip install mediapipe opencv-python")
        sys.exit(1)
    
    # Scan dataset
    print("\n📊 Scanning dataset...")
    stats = converter.scan_dataset()
    
    print(f"   Classes found: {stats['num_classes']}")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Classes: {', '.join(stats['classes'][:10])}...")
    
    # Progress callback
    def show_progress(current, total, class_name):
        pct = current / total * 100
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r   [{bar}] {pct:.1f}% - {class_name: <5}", end='', flush=True)
    
    # Convert
    max_samples = args.max_per_class
    if max_samples:
        print(f"\n⚡ Converting dataset (max {max_samples} per class)...")
    else:
        print(f"\n⚡ Converting full dataset...")
    
    success = converter.convert_dataset(
        max_per_class=max_samples,
        progress_callback=show_progress
    )
    print()  # New line after progress bar
    
    if success:
        final_stats = converter.get_statistics()
        
        print("\n✅ Preprocessing complete!")
        print(f"   Total samples: {final_stats['total_samples']}")
        print(f"   Feature dimension: {final_stats['feature_dim']}")
        print(f"   Failed images: {final_stats['failed_images']}")
        print(f"\n📁 Output saved to: {output_dir}")
        print("\n🚀 Next step: Run training with:")
        print("   python scripts/train.py")
    else:
        print("\n❌ Preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
