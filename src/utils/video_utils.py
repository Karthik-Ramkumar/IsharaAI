"""
Module: video_utils.py
Description: Webcam handling utilities
Author: Hackathon Team
Date: 2026
"""
import os
import sys
import logging
import threading
import queue
from typing import Optional, Callable
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not installed")


class VideoCapture:
    """
    Thread-safe video capture wrapper.
    
    Provides frame reading with optional threading for non-blocking operation.
    """
    
    def __init__(self, 
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        """
        Initialize video capture.
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self._capture = None
        self._is_opened = False
        self._frame_queue = queue.Queue(maxsize=2)
        self._capture_thread = None
        self._is_running = False
    
    def open(self) -> bool:
        """
        Open the video capture.
        
        Returns:
            True if successful
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV not available")
            return False
        
        try:
            self._capture = cv2.VideoCapture(self.camera_index)
            
            if not self._capture.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False
            
            # Set properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            self._is_opened = True
            logger.info(f"Camera opened: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame.
        
        Returns:
            BGR frame or None if failed
        """
        if not self._is_opened or self._capture is None:
            return None
        
        ret, frame = self._capture.read()
        return frame if ret else None
    
    def read_frame_mirrored(self) -> Optional[np.ndarray]:
        """Read frame and flip horizontally for mirror effect."""
        frame = self.read_frame()
        if frame is not None:
            return cv2.flip(frame, 1)
        return None
    
    def start_capture_thread(self) -> None:
        """Start a background thread for continuous capture."""
        if self._is_running:
            return
        
        self._is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("Capture thread started")
    
    def _capture_loop(self) -> None:
        """Background capture loop."""
        while self._is_running:
            frame = self.read_frame_mirrored()
            
            if frame is not None:
                try:
                    # Put frame in queue, drop old frames if full
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the capture thread.
        
        Returns:
            Latest frame or None if no frame available
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop_capture_thread(self) -> None:
        """Stop the background capture thread."""
        self._is_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        logger.info("Capture thread stopped")
    
    def release(self) -> None:
        """Release the video capture."""
        self.stop_capture_thread()
        
        if self._capture:
            self._capture.release()
            self._capture = None
        
        self._is_opened = False
        logger.info("Camera released")
    
    @property
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._is_opened
    
    def get_properties(self) -> dict:
        """Get camera properties."""
        if not self._capture:
            return {}
        
        return {
            'width': int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self._capture.get(cv2.CAP_PROP_FPS)),
            'backend': self._capture.getBackendName()
        }


def list_cameras(max_cameras: int = 5) -> list:
    """
    List available cameras.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List of available camera indices
    """
    if not OPENCV_AVAILABLE:
        return []
    
    available = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available


if __name__ == "__main__":
    print("Testing VideoCapture...")
    print(f"OpenCV available: {OPENCV_AVAILABLE}")
    
    if OPENCV_AVAILABLE:
        # List cameras
        cameras = list_cameras()
        print(f"\nAvailable cameras: {cameras}")
        
        if cameras:
            # Test capture
            vc = VideoCapture(cameras[0])
            
            if vc.open():
                props = vc.get_properties()
                print(f"Camera properties: {props}")
                
                print("\nCapturing 5 frames...")
                for i in range(5):
                    frame = vc.read_frame()
                    if frame is not None:
                        print(f"  Frame {i+1}: {frame.shape}")
                
                vc.release()
                print("\n✓ Test complete!")
            else:
                print("Could not open camera")
        else:
            print("No cameras found")
    else:
        print("OpenCV not installed")
