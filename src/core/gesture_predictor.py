"""
Module: gesture_predictor.py
Description: Real-time gesture prediction with smoothing
Author: Hackathon Team
Date: 2026

This module provides real-time ISL gesture prediction from
hand landmarks with temporal smoothing to reduce jitter.
"""
import os
import sys
import json
import logging
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from models.isl_classifier import ISLClassifier, TF_AVAILABLE

# Setup logging
logger = logging.getLogger(__name__)


class GesturePredictor:
    """
    Real-time gesture prediction with temporal smoothing.
    
    Uses a buffer of recent predictions to reduce jitter and
    provide more stable outputs for live recognition.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 label_map_path: Optional[str] = None,
                 buffer_size: int = 5,
                 confidence_threshold: float = 0.7):
        """
        Initialize the gesture predictor.
        
        Args:
            model_path: Path to trained model file
            label_map_path: Path to label map JSON
            buffer_size: Size of prediction buffer for smoothing
            confidence_threshold: Minimum confidence to accept prediction
        """
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        
        self.classifier = None
        self._prediction_buffer = deque(maxlen=buffer_size)
        self._current_prediction = None
        self._current_confidence = 0.0
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained model."""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
        
        # Determine paths
        if self.model_path is None:
            self.model_path = os.path.join(config.MODELS_DIR, 'isl_model.keras')
        
        if self.label_map_path is None:
            self.label_map_path = os.path.join(config.MODELS_DIR, 'label_map.json')
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found: {self.model_path}")
            return False
        
        # Load label map first to get num_classes
        num_classes = 35  # Default
        label_map = None
        
        if os.path.exists(self.label_map_path):
            with open(self.label_map_path, 'r') as f:
                label_map = json.load(f)
                num_classes = len(label_map.get('label_to_idx', {}))
        
        # Create classifier and load model
        self.classifier = ISLClassifier(num_classes=num_classes)
        
        if self.classifier.load(self.model_path, self.label_map_path):
            if label_map:
                self.classifier.set_label_map(label_map)
            logger.info(f"Model loaded: {num_classes} classes")
            return True
        
        return False
    
    @property
    def is_ready(self) -> bool:
        """Check if predictor is ready."""
        return self.classifier is not None and self.classifier.model is not None
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from landmarks (single frame).
        
        Args:
            landmarks: Feature array of shape (63,)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.is_ready:
            return "", 0.0
        
        class_name, confidence = self.classifier.predict_class(landmarks)
        return class_name, confidence
    
    def predict_smooth(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture with temporal smoothing.
        
        Uses majority voting over recent predictions to reduce jitter.
        
        Args:
            landmarks: Feature array of shape (63,)
            
        Returns:
            Tuple of (smoothed_prediction, confidence)
        """
        if not self.is_ready:
            return "", 0.0
        
        # Get raw prediction
        class_name, confidence = self.predict(landmarks)
        
        # Only add to buffer if confidence is high enough
        if confidence >= self.confidence_threshold:
            self._prediction_buffer.append((class_name, confidence))
        
        # Need enough predictions for voting
        if len(self._prediction_buffer) < 3:
            return class_name, confidence
        
        # Majority voting
        predictions = [p[0] for p in self._prediction_buffer]
        confidences = [p[1] for p in self._prediction_buffer]
        
        # Count occurrences
        from collections import Counter
        counter = Counter(predictions)
        most_common = counter.most_common(1)[0]
        
        smoothed_class = most_common[0]
        occurrence_ratio = most_common[1] / len(predictions)
        
        # Average confidence for the most common prediction
        matching_confidences = [c for p, c in self._prediction_buffer if p == smoothed_class]
        avg_confidence = np.mean(matching_confidences) if matching_confidences else 0.0
        
        # Update current state
        self._current_prediction = smoothed_class
        self._current_confidence = avg_confidence * occurrence_ratio
        
        return smoothed_class, self._current_confidence
    
    def predict_with_history(self, landmarks: np.ndarray) -> Dict:
        """
        Predict with full history information.
        
        Args:
            landmarks: Feature array
            
        Returns:
            Dictionary with prediction details
        """
        raw_class, raw_confidence = self.predict(landmarks)
        smooth_class, smooth_confidence = self.predict_smooth(landmarks)
        
        return {
            'raw_prediction': raw_class,
            'raw_confidence': raw_confidence,
            'smoothed_prediction': smooth_class,
            'smoothed_confidence': smooth_confidence,
            'buffer_size': len(self._prediction_buffer),
            'is_stable': smooth_confidence >= self.confidence_threshold
        }
    
    def reset(self) -> None:
        """Reset the prediction buffer."""
        self._prediction_buffer.clear()
        self._current_prediction = None
        self._current_confidence = 0.0
    
    def get_current_prediction(self) -> Tuple[Optional[str], float]:
        """
        Get the current stable prediction.
        
        Returns:
            Tuple of (current_prediction, confidence)
        """
        return self._current_prediction, self._current_confidence
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_buffer_size(self, size: int) -> None:
        """Set the buffer size for smoothing."""
        self.buffer_size = max(1, size)
        old_buffer = list(self._prediction_buffer)
        self._prediction_buffer = deque(old_buffer[-size:], maxlen=size)


class GesturePredictorWithDebounce(GesturePredictor):
    """
    Gesture predictor with debouncing for word building.
    
    Prevents the same gesture from being registered multiple times
    in quick succession, useful for building words from letters.
    """
    
    def __init__(self, *args, debounce_frames: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.debounce_frames = debounce_frames
        self._last_output = None
        self._frames_since_output = 0
        self._output_history = []
    
    def predict_debounced(self, landmarks: np.ndarray) -> Tuple[Optional[str], float, bool]:
        """
        Predict with debouncing.
        
        Returns:
            Tuple of (prediction, confidence, is_new)
            is_new is True only when a new stable prediction is made
        """
        smooth_class, confidence = self.predict_smooth(landmarks)
        
        self._frames_since_output += 1
        is_new = False
        
        # Check if this is a new stable prediction
        if confidence >= self.confidence_threshold:
            if smooth_class != self._last_output and self._frames_since_output >= self.debounce_frames:
                self._last_output = smooth_class
                self._frames_since_output = 0
                self._output_history.append(smooth_class)
                is_new = True
        
        return smooth_class, confidence, is_new
    
    def get_word(self) -> str:
        """Get the current word built from gestures."""
        return ''.join(self._output_history)
    
    def clear_word(self) -> None:
        """Clear the word history."""
        self._output_history.clear()
        self._last_output = None


def create_predictor(model_path: Optional[str] = None, **kwargs) -> GesturePredictor:
    """Factory function to create a gesture predictor."""
    return GesturePredictor(model_path=model_path, **kwargs)


if __name__ == "__main__":
    print("Testing GesturePredictor...")
    print(f"TensorFlow available: {TF_AVAILABLE}")
    
    if TF_AVAILABLE:
        predictor = GesturePredictor()
        
        if predictor.is_ready:
            print("\nPredictor ready!")
            
            # Test with random data
            test_landmarks = np.random.randn(63).astype(np.float32)
            prediction, confidence = predictor.predict(test_landmarks)
            print(f"Test prediction: {prediction}, confidence: {confidence:.4f}")
            
            # Test smoothing
            print("\nTesting smoothing (5 predictions):")
            for i in range(5):
                landmarks = np.random.randn(63).astype(np.float32)
                smooth_pred, smooth_conf = predictor.predict_smooth(landmarks)
                print(f"  {i+1}: {smooth_pred} ({smooth_conf:.4f})")
            
            print("\n✓ Test complete!")
        else:
            print("\nPredictor not ready - model may not be trained yet")
            print("Train with: python scripts/train.py")
    else:
        print("\nTensorFlow not installed")
        print("Install with: pip install tensorflow")
