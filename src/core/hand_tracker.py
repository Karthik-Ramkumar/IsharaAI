"""
Module: hand_tracker.py
Description: Real-time hand tracking from webcam using MediaPipe
Author: Hackathon Team
Date: 2026

This module provides real-time hand tracking and landmark extraction
for live ISL gesture recognition.
"""
import os
import sys
import logging
import numpy as np
from typing import Tuple, Optional, List

# Setup logging
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import cv2
    import mediapipe as mp
    OPENCV_AVAILABLE = True
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    MEDIAPIPE_AVAILABLE = False
    logger.warning("OpenCV or MediaPipe not installed")


class HandTracker:
    """
    Real-time hand tracking and landmark extraction.
    
    Optimized for live video processing with visualization.
    """
    
    # Landmark constants
    NUM_LANDMARKS = 21
    FEATURES_PER_LANDMARK = 3
    TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63
    
    def __init__(self,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand tracker.
        
        Args:
            max_num_hands: Maximum number of hands to track (1 for speed)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._hands = None
        self._mp_hands = None
        self._mp_drawing = None
        self._mp_drawing_styles = None
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize MediaPipe Hands for video processing."""
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not available")
            return False
        
        try:
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            
            # Use static_image_mode=False for video (faster tracking)
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            logger.info("Hand tracker initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hand tracker: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if tracker is available."""
        return MEDIAPIPE_AVAILABLE and OPENCV_AVAILABLE and self._hands is not None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a video frame and extract landmarks.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (annotated_frame, landmarks_array or None)
        """
        if not self.is_available:
            return frame.copy(), None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self._hands.process(rgb_frame)
        
        # Draw landmarks on frame
        annotated = frame.copy()
        landmarks = None
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extract landmarks from first hand
            landmarks = self._extract_and_normalize(results.multi_hand_landmarks[0])
        
        return annotated, landmarks
    
    def _extract_and_normalize(self, hand_landmarks) -> np.ndarray:
        """
        Extract and normalize landmarks from MediaPipe result.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Normalized landmarks array of shape (63,)
        """
        # Extract raw landmarks
        raw = np.zeros((self.NUM_LANDMARKS, self.FEATURES_PER_LANDMARK))
        for i, lm in enumerate(hand_landmarks.landmark):
            raw[i] = [lm.x, lm.y, lm.z]
        
        # Normalize relative to wrist
        wrist = raw[0].copy()
        normalized = raw - wrist
        
        # Scale by palm size
        scale = np.linalg.norm(raw[9] - raw[0])
        if scale > 0:
            normalized = normalized / scale
        
        return normalized.flatten()
    
    def get_hand_detected(self, frame: np.ndarray) -> bool:
        """
        Check if a hand is detected in the frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            True if hand detected
        """
        if not self.is_available:
            return False
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        return results.multi_hand_landmarks is not None
    
    def get_landmarks_only(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get just the landmarks without drawing.
        
        More efficient when visualization is not needed.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Landmarks array or None
        """
        if not self.is_available:
            return None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        
        if results.multi_hand_landmarks:
            return self._extract_and_normalize(results.multi_hand_landmarks[0])
        
        return None
    
    def close(self) -> None:
        """Release resources."""
        if self._hands:
            self._hands.close()


class HandTrackerWithROI(HandTracker):
    """
    Hand tracker with Region of Interest (ROI) optimization.
    
    Tracks where the hand was last seen and focuses detection
    on that region for faster processing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_roi = None
        self._roi_padding = 50
    
    def process_frame_with_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple]]:
        """
        Process frame with ROI tracking.
        
        Returns:
            Tuple of (annotated_frame, landmarks, roi_box)
        """
        annotated, landmarks = self.process_frame(frame)
        
        # Update ROI based on detected hand
        if landmarks is not None:
            # Calculate bounding box from landmarks
            # (For simplicity, using full frame - could optimize)
            self._last_roi = (0, 0, frame.shape[1], frame.shape[0])
        
        return annotated, landmarks, self._last_roi


def create_tracker(**kwargs) -> HandTracker:
    """Factory function to create a hand tracker."""
    return HandTracker(**kwargs)


if __name__ == "__main__":
    print("Testing HandTracker...")
    print(f"OpenCV available: {OPENCV_AVAILABLE}")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    
    if OPENCV_AVAILABLE and MEDIAPIPE_AVAILABLE:
        tracker = HandTracker()
        
        if tracker.is_available:
            print("\nStarting webcam test (press 'q' to quit)...")
            
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated, landmarks = tracker.process_frame(frame)
                
                # Show status
                status = "Hand detected!" if landmarks is not None else "No hand"
                cv2.putText(annotated, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if landmarks is not None:
                    cv2.putText(annotated, f"Features: {len(landmarks)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Hand Tracker Test', annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            tracker.close()
            
            print("\n✓ Test complete!")
        else:
            print("Tracker not available")
    else:
        print("\nInstall dependencies:")
        print("  pip install opencv-python mediapipe")
