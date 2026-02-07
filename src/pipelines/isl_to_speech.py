"""
Module: isl_to_speech.py
Description: Pipeline 3: ISL → Text → Speech
Author: Hackathon Team
Date: 2026

This pipeline converts live ISL hand gestures into spoken audio.
It combines hand tracking, gesture prediction, and text-to-speech.
"""
import os
import sys
import logging
import threading
import queue
from typing import Dict, Optional, Callable, Generator
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.core.hand_tracker import HandTracker, OPENCV_AVAILABLE
from src.core.gesture_predictor import GesturePredictor, GesturePredictorWithDebounce
from src.core.text_to_speech import TextToSpeech, PYTTSX3_AVAILABLE

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
except ImportError:
    cv2 = None


class ISLToSpeechPipeline:
    """
    Pipeline for converting ISL gestures to speech.
    
    This pipeline:
    1. Captures video from webcam
    2. Tracks hand and extracts landmarks
    3. Predicts gesture using trained model
    4. Converts predicted text to speech
    
    Example usage:
        pipeline = ISLToSpeechPipeline()
        if pipeline.is_ready():
            for result in pipeline.start_stream():
                print(f"Detected: {result['gesture']}")
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 speak_enabled: bool = True,
                 debounce_frames: int = 15):
        """
        Initialize the ISL to Speech pipeline.
        
        Args:
            model_path: Path to trained gesture model
            speak_enabled: Whether to enable TTS output
            debounce_frames: Frames to wait before accepting new gesture
        """
        self.speak_enabled = speak_enabled
        self.debounce_frames = debounce_frames
        
        # Initialize components
        self.hand_tracker = None
        self.gesture_predictor = None
        self.tts = None
        
        # State
        self._last_spoken = None
        self._word_buffer = []
        self._is_running = False
        self._video_capture = None
        
        # Initialize
        self._initialize(model_path)
        
        logger.info("ISLToSpeechPipeline initialized")
    
    def _initialize(self, model_path: Optional[str]) -> None:
        """Initialize all pipeline components."""
        # Hand tracker
        self.hand_tracker = HandTracker(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture predictor with debouncing
        self.gesture_predictor = GesturePredictorWithDebounce(
            model_path=model_path,
            buffer_size=config.PREDICTION_SMOOTHING_WINDOW,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            debounce_frames=self.debounce_frames
        )
        
        # Text-to-speech
        if self.speak_enabled and PYTTSX3_AVAILABLE:
            self.tts = TextToSpeech(
                rate=config.SPEECH_RATE,
                volume=config.SPEECH_VOLUME
            )
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready."""
        tracker_ready = self.hand_tracker and self.hand_tracker.is_available
        predictor_ready = self.gesture_predictor and self.gesture_predictor.is_ready
        
        if not tracker_ready:
            logger.warning("Hand tracker not available")
        if not predictor_ready:
            logger.warning("Gesture predictor not ready - model may not be trained")
        
        return tracker_ready and predictor_ready
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single video frame.
        
        Args:
            frame: BGR video frame
            
        Returns:
            Dictionary with:
                - annotated_frame: Frame with landmarks drawn
                - gesture: Detected gesture (or None)
                - confidence: Prediction confidence
                - is_new: Whether this is a new gesture
                - word: Current accumulated word
                - spoken: Whether text was spoken
        """
        result = {
            'annotated_frame': frame.copy(),
            'gesture': None,
            'confidence': 0.0,
            'is_new': False,
            'word': '',
            'spoken': False
        }
        
        if not self.hand_tracker or not self.gesture_predictor:
            return result
        
        # Track hand and get landmarks
        annotated, landmarks = self.hand_tracker.process_frame(frame)
        result['annotated_frame'] = annotated
        
        if landmarks is None:
            return result
        
        # Predict gesture with debouncing
        gesture, confidence, is_new = self.gesture_predictor.predict_debounced(landmarks)
        
        result['gesture'] = gesture
        result['confidence'] = confidence
        result['is_new'] = is_new
        result['word'] = self.gesture_predictor.get_word()
        
        # Speak new gestures
        if is_new and self.speak_enabled and self.tts:
            self.tts.speak_async(gesture)
            result['spoken'] = True
            logger.info(f"Detected and spoke: {gesture}")
        
        return result
    
    def start_stream(self, camera_index: int = 0) -> Generator[Dict, None, None]:
        """
        Start streaming video and yield results.
        
        Args:
            camera_index: Camera device index
            
        Yields:
            Result dictionary for each frame
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV not available")
            return
        
        self._video_capture = cv2.VideoCapture(camera_index)
        
        if not self._video_capture.isOpened():
            logger.error("Could not open camera")
            return
        
        # Set resolution
        self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        self._is_running = True
        logger.info("Started video stream")
        
        try:
            while self._is_running:
                ret, frame = self._video_capture.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                result = self.process_frame(frame)
                
                yield result
                
        finally:
            self.stop_stream()
    
    def stop_stream(self) -> None:
        """Stop the video stream."""
        self._is_running = False
        
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
        
        logger.info("Stopped video stream")
    
    def get_current_word(self) -> str:
        """Get the current accumulated word."""
        if self.gesture_predictor:
            return self.gesture_predictor.get_word()
        return ""
    
    def clear_word(self) -> None:
        """Clear the accumulated word."""
        if self.gesture_predictor:
            self.gesture_predictor.clear_word()
    
    def speak_word(self) -> None:
        """Speak the current accumulated word."""
        word = self.get_current_word()
        if word and self.tts:
            self.tts.speak_async(word)
            logger.info(f"Spoke word: {word}")
    
    def reset(self) -> None:
        """Reset the pipeline state."""
        if self.gesture_predictor:
            self.gesture_predictor.reset()
            self.gesture_predictor.clear_word()
        self._last_spoken = None
    
    def set_speak_enabled(self, enabled: bool) -> None:
        """Enable or disable speech output."""
        self.speak_enabled = enabled
    
    def get_status(self) -> Dict:
        """Get pipeline status."""
        return {
            'tracker_available': self.hand_tracker.is_available if self.hand_tracker else False,
            'predictor_ready': self.gesture_predictor.is_ready if self.gesture_predictor else False,
            'tts_available': self.tts.is_available if self.tts else False,
            'is_running': self._is_running,
            'current_word': self.get_current_word(),
            'speak_enabled': self.speak_enabled
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_stream()
        
        if self.hand_tracker:
            self.hand_tracker.close()
        
        if self.tts:
            self.tts.shutdown()


def create_pipeline(**kwargs) -> ISLToSpeechPipeline:
    """Factory function to create an ISL to Speech pipeline."""
    return ISLToSpeechPipeline(**kwargs)


def run_demo():
    """Run a demo of the pipeline."""
    print("ISL to Speech Pipeline Demo")
    print("=" * 40)
    
    pipeline = ISLToSpeechPipeline()
    
    status = pipeline.get_status()
    print(f"\nPipeline Status:")
    print(f"  Hand Tracker: {'✓' if status['tracker_available'] else '✗'}")
    print(f"  Gesture Predictor: {'✓' if status['predictor_ready'] else '✗'}")
    print(f"  Text-to-Speech: {'✓' if status['tts_available'] else '✗'}")
    
    if not pipeline.is_ready():
        print("\n⚠ Pipeline not fully ready!")
        if not status['predictor_ready']:
            print("  Train model first: python scripts/train.py")
        return
    
    print("\nStarting webcam (press 'q' to quit, 'c' to clear word, 's' to speak word)")
    
    for result in pipeline.start_stream():
        frame = result['annotated_frame']
        
        # Draw status
        gesture = result['gesture'] or "No gesture"
        confidence = result['confidence']
        word = result['word']
        
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Word: {word}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        if result['is_new']:
            cv2.putText(frame, "NEW!", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('ISL to Speech', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            pipeline.clear_word()
        elif key == ord('s'):
            pipeline.speak_word()
    
    cv2.destroyAllWindows()
    pipeline.cleanup()
    print("\nDemo complete!")


if __name__ == "__main__":
    run_demo()
