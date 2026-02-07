"""
Module: isl_classifier.py
Description: Neural network for ISL gesture classification
Author: Hackathon Team
Date: 2026

This module provides a Dense neural network for classifying ISL gestures
from hand landmark features.
"""
import os
import sys
import json
import logging
import numpy as np
from typing import Tuple, Optional, Dict, List

# Setup logging
logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. Model training unavailable.")


class ISLClassifier:
    """
    Neural network classifier for ISL gesture recognition.
    
    Uses a Dense neural network architecture optimized for
    hand landmark features (63 dimensions).
    """
    
    def __init__(self, num_classes: int, input_shape: Tuple[int] = (63,)):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of ISL signs to classify
            input_shape: Shape of input features
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.label_map = None
        self.reverse_label_map = None
        
        if TF_AVAILABLE:
            self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """
        Build the neural network model.
        
        Architecture:
        - Input: (63,) landmarks
        - Dense(256, relu) + BatchNorm + Dropout(0.3)
        - Dense(128, relu) + BatchNorm + Dropout(0.3)
        - Dense(64, relu) + BatchNorm + Dropout(0.2)
        - Dense(num_classes, softmax)
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Hidden layer 1
            layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden layer 2
            layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden layer 3
            layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            return
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model compiled successfully")
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: np.ndarray, 
             y_val: np.ndarray,
             epochs: int = 50,
             batch_size: int = 32,
             model_save_path: Optional[str] = None) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            model_save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            logger.error("Model not available")
            return {}
        
        # Compile if not already
        if self.model.optimizer is None:
            self.compile_model()
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        if model_save_path:
            callback_list.append(
                callbacks.ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        logger.info(f"Starting training: {len(X_train)} samples, {epochs} epochs")
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        # Log final metrics
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        logger.info(f"Training complete: acc={final_acc:.4f}, val_acc={final_val_acc:.4f}")
        
        return {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    
    def predict(self, landmarks: np.ndarray) -> Tuple[int, float]:
        """
        Predict the class for a single sample.
        
        Args:
            landmarks: Feature array of shape (63,)
            
        Returns:
            Tuple of (predicted_class_idx, confidence)
        """
        if self.model is None:
            return -1, 0.0
        
        # Ensure correct shape
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(landmarks, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return int(class_idx), confidence
    
    def predict_class(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class name for a sample.
        
        Args:
            landmarks: Feature array of shape (63,)
            
        Returns:
            Tuple of (class_name, confidence)
        """
        class_idx, confidence = self.predict(landmarks)
        
        if self.reverse_label_map and class_idx in self.reverse_label_map:
            class_name = self.reverse_label_map[class_idx]
        else:
            class_name = str(class_idx)
        
        return class_name, confidence
    
    def predict_batch(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes for multiple samples.
        
        Args:
            landmarks: Feature array of shape (N, 63)
            
        Returns:
            Tuple of (class_indices, confidences)
        """
        if self.model is None:
            return np.array([]), np.array([])
        
        predictions = self.model.predict(landmarks, verbose=0)
        class_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return class_indices, confidences
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            return {}
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for confusion matrix
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'y_pred': y_pred,
            'y_true': y_test
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            return
        
        self.model.save(path)
        
        # Save label map if available
        if self.label_map:
            map_path = path.replace('.h5', '_labels.json').replace('.keras', '_labels.json')
            with open(map_path, 'w') as f:
                json.dump({
                    'label_to_idx': self.label_map,
                    'idx_to_label': {str(k): v for k, v in self.reverse_label_map.items()}
                }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str, label_map_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the model file
            label_map_path: Optional path to label map JSON
            
        Returns:
            True if successful
        """
        if not TF_AVAILABLE:
            return False
        
        try:
            self.model = keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
            
            # Try to load label map
            if label_map_path and os.path.exists(label_map_path):
                with open(label_map_path, 'r') as f:
                    maps = json.load(f)
                    self.label_map = maps.get('label_to_idx', {})
                    self.reverse_label_map = {int(k): v for k, v in maps.get('idx_to_label', {}).items()}
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def set_label_map(self, label_map: Dict) -> None:
        """
        Set the label map for class name lookup.
        
        Args:
            label_map: Dictionary with 'label_to_idx' and 'idx_to_label'
        """
        self.label_map = label_map.get('label_to_idx', label_map)
        if 'idx_to_label' in label_map:
            self.reverse_label_map = {int(k): v for k, v in label_map['idx_to_label'].items()}
        else:
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "Model not available"
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


def create_classifier(num_classes: int, input_shape: Tuple[int] = (63,)) -> ISLClassifier:
    """Factory function to create a classifier."""
    return ISLClassifier(num_classes, input_shape)


if __name__ == "__main__":
    print("Testing ISLClassifier...")
    print(f"TensorFlow available: {TF_AVAILABLE}")
    
    if TF_AVAILABLE:
        # Create a test classifier
        classifier = ISLClassifier(num_classes=35, input_shape=(63,))
        classifier.compile_model()
        
        print("\nModel Summary:")
        print(classifier.summary())
        
        # Test with random data
        X_test = np.random.randn(10, 63).astype(np.float32)
        y_test = np.random.randint(0, 35, 10)
        
        # Test prediction
        class_idx, confidence = classifier.predict(X_test[0])
        print(f"\nTest prediction: class={class_idx}, confidence={confidence:.4f}")
        
        print("\n✓ Classifier test complete!")
    else:
        print("TensorFlow not installed. Install with: pip install tensorflow")
