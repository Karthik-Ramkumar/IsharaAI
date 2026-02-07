"""
Module: train_model.py
Description: Training pipeline for ISL gesture classifier
Author: Hackathon Team
Date: 2026

This module provides functions for training and evaluating
the ISL gesture recognition model.
"""
import os
import sys
import json
import logging
import numpy as np
from typing import Tuple, Optional, Dict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from models.isl_classifier import ISLClassifier, TF_AVAILABLE

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import sklearn for data splitting
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_data(data_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """
    Load preprocessed landmark data.
    
    Args:
        data_dir: Directory containing X.npy, y.npy, and label_map.json
        
    Returns:
        Tuple of (X, y, label_map) or (None, None, None) if not found
    """
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')
    map_path = os.path.join(data_dir, 'label_map.json')
    
    if not all(os.path.exists(p) for p in [X_path, y_path, map_path]):
        logger.error(f"Data files not found in {data_dir}")
        logger.info("Run preprocessing first: python scripts/preprocess_dataset.py")
        return None, None, None
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    with open(map_path, 'r') as f:
        label_map = json.load(f)
    
    logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, label_map


def prepare_data(X: np.ndarray, y: np.ndarray, 
                test_size: float = 0.2,
                val_size: float = 0.1,
                random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature array
        y: Label array
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test splits
    """
    if not SKLEARN_AVAILABLE:
        logger.error("sklearn not available for data splitting")
        return {}
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def train_classifier(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    num_classes: int,
                    epochs: int = 50,
                    batch_size: int = 32,
                    save_path: Optional[str] = None) -> Tuple[ISLClassifier, Dict]:
    """
    Train the ISL classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        num_classes: Number of classes
        epochs: Training epochs
        batch_size: Batch size
        save_path: Path to save the best model
        
    Returns:
        Tuple of (trained_classifier, training_history)
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None, {}
    
    # Create classifier
    classifier = ISLClassifier(num_classes=num_classes, input_shape=(X_train.shape[1],))
    classifier.compile_model()
    
    logger.info("Starting training...")
    
    # Train
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=save_path
    )
    
    return classifier, history


def evaluate_model(classifier: ISLClassifier, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  label_map: Dict) -> Dict:
    """
    Evaluate the trained model.
    
    Args:
        classifier: Trained classifier
        X_test: Test features
        y_test: Test labels
        label_map: Label mapping dictionary
        
    Returns:
        Evaluation results
    """
    results = classifier.evaluate(X_test, y_test)
    
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test loss: {results['loss']:.4f}")
    
    # Classification report
    if SKLEARN_AVAILABLE:
        idx_to_label = {int(k): v for k, v in label_map.get('idx_to_label', {}).items()}
        target_names = [idx_to_label.get(i, str(i)) for i in range(len(idx_to_label))]
        
        report = classification_report(
            results['y_true'], 
            results['y_pred'],
            target_names=target_names,
            output_dict=True
        )
        results['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        results['confusion_matrix'] = cm.tolist()
    
    return results


def save_training_results(save_dir: str, 
                         history: Dict,
                         eval_results: Dict,
                         label_map: Dict) -> None:
    """
    Save training results to disk.
    
    Args:
        save_dir: Directory to save results
        history: Training history
        eval_results: Evaluation results
        label_map: Label mapping
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        # Convert numpy arrays to lists
        history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_serializable, f, indent=2)
    
    # Save evaluation results
    eval_serializable = {
        'accuracy': eval_results['accuracy'],
        'loss': eval_results['loss']
    }
    if 'classification_report' in eval_results:
        eval_serializable['classification_report'] = eval_results['classification_report']
    if 'confusion_matrix' in eval_results:
        eval_serializable['confusion_matrix'] = eval_results['confusion_matrix']
    
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_serializable, f, indent=2)
    
    # Save label map
    with open(os.path.join(save_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)
    
    logger.info(f"Results saved to {save_dir}")


def full_training_pipeline(data_dir: str = None,
                          model_dir: str = None,
                          epochs: int = 50,
                          batch_size: int = 32) -> bool:
    """
    Run the full training pipeline.
    
    Args:
        data_dir: Directory with preprocessed data
        model_dir: Directory to save trained model
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        True if successful
    """
    # Set default paths
    if data_dir is None:
        data_dir = config.LANDMARKS_DIR
    if model_dir is None:
        model_dir = config.MODELS_DIR
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    X, y, label_map = load_data(data_dir)
    if X is None:
        return False
    
    print(f"   Loaded {len(X)} samples, {len(set(y))} classes")
    
    # Prepare data splits
    print("\n2. Preparing data splits...")
    splits = prepare_data(X, y)
    if not splits:
        return False
    
    print(f"   Train: {len(splits['X_train'])}")
    print(f"   Validation: {len(splits['X_val'])}")
    print(f"   Test: {len(splits['X_test'])}")
    
    # Train
    print("\n3. Training model...")
    model_path = os.path.join(model_dir, 'isl_model.keras')
    
    classifier, history = train_classifier(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        num_classes=len(set(y)),
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_path
    )
    
    if classifier is None:
        return False
    
    # Set label map
    classifier.set_label_map(label_map)
    
    # Evaluate
    print("\n4. Evaluating model...")
    eval_results = evaluate_model(classifier, splits['X_test'], splits['y_test'], label_map)
    
    print(f"\n   Test Accuracy: {eval_results['accuracy']*100:.2f}%")
    print(f"   Test Loss: {eval_results['loss']:.4f}")
    
    # Save results
    print("\n5. Saving results...")
    save_training_results(model_dir, history, eval_results, label_map)
    classifier.save(model_path)
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Final accuracy: {eval_results['accuracy']*100:.2f}%")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ISL gesture classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--model-dir', type=str, default=None, help='Model save directory')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ISL Gesture Classifier Training")
    print("=" * 50)
    
    if not TF_AVAILABLE:
        print("\nError: TensorFlow not installed")
        print("Install with: pip install tensorflow")
        sys.exit(1)
    
    if not SKLEARN_AVAILABLE:
        print("\nError: scikit-learn not available")
        print("Install with: pip install scikit-learn")
        sys.exit(1)
    
    success = full_training_pipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    sys.exit(0 if success else 1)
