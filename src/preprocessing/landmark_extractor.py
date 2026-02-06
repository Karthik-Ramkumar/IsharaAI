"""
Module: landmark_extractor.py
Description: Extract MediaPipe hand landmarks from images
Author: Hackathon Team
Date: 2026

This module provides hand landmark extraction using MediaPipe,
converting images of hand gestures to normalized landmark coordinates.
"""
import os
import sys
import logging
import numpy as np
from typing import Optional, Tuple, List

# Setup logging
logger = logging.getLogger(__name__)

# Try to import MediaPipe
try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe or OpenCV not installed. Landmark extraction unavailable.")


class LandmarkExtractor:
    """
    Extract hand landmarks from images using MediaPipe.
    
    Converts images of hand gestures to normalized 63-dimensional
    landmark vectors (21 landmarks × 3 coordinates).
    """
    
    # Number of hand landmarks
    NUM_LANDMARKS = 21
    # Features per landmark (x, y, z)
    FEATURES_PER_LANDMARK = 3
    # Total features
    TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the landmark extractor.
        
        Args:
            static_image_mode: If True, treats each image independently (for dataset processing)
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._hands = None
        self._mp_hands = None
        self._mp_drawing = None
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize MediaPipe Hands."""
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not available")
            return False
        
        try:
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            
            self._hands = self._mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            logger.info("MediaPipe Hands initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if landmark extraction is available."""
        return MEDIAPIPE_AVAILABLE and self._hands is not None
    
    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of shape (63,) containing normalized landmarks,
            or None if no hand detected
        """
        if not self.is_available:
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            return self.extract_from_frame(image)
            
        except Exception as e:
            logger.error(f"Error extracting from {image_path}: {e}")
            return None
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a video frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            numpy array of shape (63,) containing normalized landmarks,
            or None if no hand detected
        """
        if not self.is_available:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self._hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
            
            # Get landmarks from first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract and normalize landmarks
            landmarks = self._extract_landmarks(hand_landmarks)
            normalized = self._normalize_landmarks(landmarks)
            
            return normalized.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Extract raw landmark coordinates.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            numpy array of shape (21, 3)
        """
        landmarks = np.zeros((self.NUM_LANDMARKS, self.FEATURES_PER_LANDMARK))
        
        for i, lm in enumerate(hand_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
        
        return landmarks
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position.
        
        This makes the model robust to hand position and size.
        
        Args:
            landmarks: Raw landmarks of shape (21, 3)
            
        Returns:
            Normalized landmarks of shape (21, 3)
        """
        # Get wrist position (landmark 0)
        wrist = landmarks[0].copy()
        
        # Subtract wrist position from all landmarks
        normalized = landmarks - wrist
        
        # Scale to normalize hand size
        # Use distance from wrist to middle finger MCP (landmark 9) as reference
        scale = np.linalg.norm(landmarks[9] - landmarks[0])
        if scale > 0:
            normalized = normalized / scale
        
        return normalized
    
    def extract_batch(self, image_paths: List[str], 
                     progress_callback: Optional[callable] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Extract landmarks from multiple images.
        
        Args:
            image_paths: List of image paths
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Tuple of (landmarks array, list of successfully processed indices)
        """
        landmarks_list = []
        valid_indices = []
        
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            landmarks = self.extract_from_image(path)
            
            if landmarks is not None:
                landmarks_list.append(landmarks)
                valid_indices.append(i)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        if landmarks_list:
            return np.array(landmarks_list), valid_indices
        
        return np.array([]), []
    
    def visualize_landmarks(self, image: np.ndarray, 
                           landmarks_result=None) -> np.ndarray:
        """
        Draw landmarks on an image for visualization.
        
        Args:
            image: Input image (BGR)
            landmarks_result: MediaPipe results (if None, will process image)
            
        Returns:
            Image with landmarks drawn
        """
        if not self.is_available:
            return image.copy()
        
        annotated = image.copy()
        
        if landmarks_result is None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks_result = self._hands.process(rgb)
        
        if landmarks_result.multi_hand_landmarks:
            for hand_landmarks in landmarks_result.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS
                )
        
        return annotated
    
    def process_frame_with_visualization(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a frame and return both visualization and landmarks.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (annotated_frame, landmarks_array or None)
        """
        if not self.is_available:
            return frame.copy(), None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        
        # Draw landmarks
        annotated = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS
                )
        
        # Extract landmarks
        landmarks = None
        if results.multi_hand_landmarks:
            raw = self._extract_landmarks(results.multi_hand_landmarks[0])
            normalized = self._normalize_landmarks(raw)
            landmarks = normalized.flatten()
        
        return annotated, landmarks
    
    def close(self) -> None:
        """Release resources."""
        if self._hands:
            self._hands.close()


def create_extractor(**kwargs) -> LandmarkExtractor:
    """Factory function to create a LandmarkExtractor."""
    return LandmarkExtractor(**kwargs)


if __name__ == "__main__":
    print("Testing LandmarkExtractor...")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    
    if MEDIAPIPE_AVAILABLE:
        extractor = LandmarkExtractor()
        
        if extractor.is_available:
            print(f"Extractor ready")
            print(f"Features per sample: {extractor.TOTAL_FEATURES}")
            
            # Test with sample image if available
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            try:
                import config
                test_dir = os.path.join(config.RAW_DATA_DIR, 'A')
                if os.path.exists(test_dir):
                    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:1]
                    if images:
                        test_path = os.path.join(test_dir, images[0])
                        print(f"\nTesting with: {test_path}")
                        landmarks = extractor.extract_from_image(test_path)
                        if landmarks is not None:
                            print(f"Extracted landmarks shape: {landmarks.shape}")
                            print(f"Sample values: {landmarks[:6]}")
                        else:
                            print("No hand detected in image")
            except Exception as e:
                print(f"Test error: {e}")
        else:
            print("Extractor not available")
    else:
        print("MediaPipe not installed. Install with: pip install mediapipe opencv-python")
