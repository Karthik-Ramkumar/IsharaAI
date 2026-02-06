"""
Module: dataset_converter.py
Description: Convert ISL image dataset to landmark format for training
Author: Hackathon Team
Date: 2026

This module processes the raw ISL image dataset and converts all images
to normalized landmark vectors suitable for ML model training.
"""
import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.landmark_extractor import LandmarkExtractor, MEDIAPIPE_AVAILABLE

# Setup logging
logger = logging.getLogger(__name__)


class DatasetConverter:
    """
    Converts raw ISL image dataset to landmark arrays.
    
    Processes all class folders, extracts landmarks from each image,
    and saves the results in numpy format for training.
    """
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        """
        Initialize the dataset converter.
        
        Args:
            raw_data_dir: Directory containing raw ISL images organized by class
            output_dir: Directory to save processed landmark data
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        
        self.extractor = None
        self.label_map = {}  # class_name -> index
        self.reverse_label_map = {}  # index -> class_name
        
        self._stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'classes': [],
            'samples_per_class': {}
        }
        
        # Initialize extractor
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the landmark extractor."""
        if MEDIAPIPE_AVAILABLE:
            self.extractor = LandmarkExtractor(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            logger.info("Dataset converter initialized")
        else:
            logger.error("MediaPipe not available - cannot convert dataset")
    
    @property
    def is_available(self) -> bool:
        """Check if converter is ready."""
        return self.extractor is not None and self.extractor.is_available
    
    def scan_dataset(self) -> Dict:
        """
        Scan the raw dataset and collect statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not os.path.exists(self.raw_data_dir):
            logger.error(f"Dataset directory not found: {self.raw_data_dir}")
            return {}
        
        classes = []
        samples_per_class = {}
        total_images = 0
        
        # Get all class folders
        for item in sorted(os.listdir(self.raw_data_dir)):
            class_dir = os.path.join(self.raw_data_dir, item)
            if not os.path.isdir(class_dir):
                continue
            
            # Count images in this class
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if images:
                classes.append(item)
                samples_per_class[item] = len(images)
                total_images += len(images)
        
        self._stats['classes'] = classes
        self._stats['samples_per_class'] = samples_per_class
        self._stats['total_images'] = total_images
        
        # Create label map
        self.label_map = {name: idx for idx, name in enumerate(classes)}
        self.reverse_label_map = {idx: name for name, idx in self.label_map.items()}
        
        logger.info(f"Dataset scan complete: {len(classes)} classes, {total_images} images")
        
        return {
            'num_classes': len(classes),
            'total_images': total_images,
            'classes': classes,
            'samples_per_class': samples_per_class
        }
    
    def convert_dataset(self, 
                       max_per_class: Optional[int] = None,
                       progress_callback: Optional[callable] = None) -> bool:
        """
        Convert the entire dataset to landmarks.
        
        Args:
            max_per_class: Maximum samples per class (for faster testing)
            progress_callback: Optional callback(processed, total, class_name)
            
        Returns:
            True if successful
        """
        if not self.is_available:
            logger.error("Converter not available")
            return False
        
        # Scan dataset first
        stats = self.scan_dataset()
        if not stats:
            return False
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        all_landmarks = []
        all_labels = []
        
        total_images = stats['total_images']
        processed = 0
        
        logger.info(f"Starting conversion of {total_images} images...")
        
        for class_name in stats['classes']:
            class_dir = os.path.join(self.raw_data_dir, class_name)
            class_idx = self.label_map[class_name]
            
            # Get image files
            images = sorted([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            
            if max_per_class:
                images = images[:max_per_class]
            
            class_landmarks = []
            class_failed = 0
            
            for img_file in images:
                img_path = os.path.join(class_dir, img_file)
                
                # Extract landmarks
                landmarks = self.extractor.extract_from_image(img_path)
                
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    all_labels.append(class_idx)
                    class_landmarks.append(landmarks)
                else:
                    class_failed += 1
                    self._stats['failed_images'] += 1
                
                processed += 1
                self._stats['processed_images'] = processed
                
                if progress_callback:
                    progress_callback(processed, total_images, class_name)
            
            logger.info(f"Class '{class_name}': {len(class_landmarks)} processed, {class_failed} failed")
        
        # Convert to numpy arrays
        X = np.array(all_landmarks)
        y = np.array(all_labels)
        
        # Save arrays
        np.save(os.path.join(self.output_dir, 'X.npy'), X)
        np.save(os.path.join(self.output_dir, 'y.npy'), y)
        
        # Save label map
        with open(os.path.join(self.output_dir, 'label_map.json'), 'w') as f:
            json.dump({
                'label_to_idx': self.label_map,
                'idx_to_label': {str(k): v for k, v in self.reverse_label_map.items()}
            }, f, indent=2)
        
        # Save statistics
        final_stats = {
            'total_samples': len(X),
            'num_classes': len(self.label_map),
            'feature_dim': X.shape[1] if len(X) > 0 else 0,
            'samples_per_class': {name: int(np.sum(y == idx)) 
                                  for name, idx in self.label_map.items()},
            'failed_images': self._stats['failed_images']
        }
        
        with open(os.path.join(self.output_dir, 'stats.json'), 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f"Conversion complete! {len(X)} samples saved to {self.output_dir}")
        
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats_file = os.path.join(self.output_dir, 'stats.json')
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                return json.load(f)
        
        return self._stats
    
    def load_processed_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Load previously processed data.
        
        Returns:
            Tuple of (X, y, label_map) or (None, None, None) if not found
        """
        X_path = os.path.join(self.output_dir, 'X.npy')
        y_path = os.path.join(self.output_dir, 'y.npy')
        map_path = os.path.join(self.output_dir, 'label_map.json')
        
        if not all(os.path.exists(p) for p in [X_path, y_path, map_path]):
            logger.warning("Processed data not found")
            return None, None, None
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        with open(map_path, 'r') as f:
            label_map = json.load(f)
        
        logger.info(f"Loaded {len(X)} samples with {len(label_map['label_to_idx'])} classes")
        
        return X, y, label_map


def convert_dataset(raw_dir: str, output_dir: str, 
                   max_per_class: Optional[int] = None) -> bool:
    """
    Convenience function to convert a dataset.
    
    Args:
        raw_dir: Path to raw image dataset
        output_dir: Path to save processed data
        max_per_class: Optional limit on samples per class
        
    Returns:
        True if successful
    """
    converter = DatasetConverter(raw_dir, output_dir)
    
    def progress(current, total, class_name):
        if current % 100 == 0:
            print(f"Progress: {current}/{total} ({class_name})")
    
    return converter.convert_dataset(max_per_class, progress)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ISL dataset to landmarks')
    parser.add_argument('--max-per-class', type=int, default=None,
                       help='Maximum samples per class (for testing)')
    args = parser.parse_args()
    
    print("Testing DatasetConverter...")
    
    try:
        import config
        
        converter = DatasetConverter(config.RAW_DATA_DIR, config.LANDMARKS_DIR)
        
        if not converter.is_available:
            print("Converter not available - MediaPipe may not be installed")
            sys.exit(1)
        
        # Scan dataset
        print("\nScanning dataset...")
        stats = converter.scan_dataset()
        print(f"Found {stats['num_classes']} classes with {stats['total_images']} images")
        print(f"Classes: {stats['classes']}")
        
        # Convert (limit samples for testing)
        max_samples = args.max_per_class or 50  # Default to 50 per class for testing
        print(f"\nConverting dataset (max {max_samples} per class)...")
        
        def show_progress(current, total, class_name):
            if current % 50 == 0:
                pct = current / total * 100
                print(f"  {current}/{total} ({pct:.1f}%) - Processing: {class_name}")
        
        success = converter.convert_dataset(max_per_class=max_samples, 
                                           progress_callback=show_progress)
        
        if success:
            print("\n✓ Conversion complete!")
            final_stats = converter.get_statistics()
            print(f"  Total samples: {final_stats['total_samples']}")
            print(f"  Number of classes: {final_stats['num_classes']}")
            print(f"  Feature dimension: {final_stats['feature_dim']}")
            print(f"  Failed images: {final_stats['failed_images']}")
        else:
            print("\n✗ Conversion failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
