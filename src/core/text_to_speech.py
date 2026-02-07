"""
Module: text_to_speech.py
Description: Text-to-speech synthesis using pyttsx3
Author: Hackathon Team
Date: 2026

This module provides text-to-speech capabilities for converting
recognized ISL gestures into spoken audio.
"""
import os
import sys
import logging
import threading
import queue
from typing import Optional, Callable

# Setup logging
logger = logging.getLogger(__name__)

# Try to import pyttsx3
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed. Text-to-speech will be unavailable.")


class TextToSpeech:
    """
    Text-to-speech engine using pyttsx3.
    
    Provides both synchronous and asynchronous speech synthesis.
    Uses a background thread to prevent UI blocking.
    """
    
    def __init__(self, rate: int = 150, volume: float = 0.9):
        """
        Initialize the TTS engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        self._engine = None
        self._is_initialized = False
        self._speech_queue = queue.Queue()
        self._is_speaking = False
        self._stop_requested = False
        
        # Start background thread for async speech
        self._worker_thread = None
        
        # Initialize engine
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize the pyttsx3 engine.
        
        Returns:
            True if successful, False otherwise
        """
        if not PYTTSX3_AVAILABLE:
            logger.error("pyttsx3 library not available")
            return False
        
        try:
            self._engine = pyttsx3.init()
            
            # Configure voice settings
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            
            # Try to select a good voice
            voices = self._engine.getProperty('voices')
            if voices:
                # Prefer female voice if available (often clearer)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self._engine.setProperty('voice', voice.id)
                        logger.info(f"Selected voice: {voice.name}")
                        break
            
            self._is_initialized = True
            logger.info("Text-to-speech engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return PYTTSX3_AVAILABLE and self._is_initialized
    
    def speak(self, text: str) -> None:
        """
        Speak text synchronously (blocks until complete).
        
        Args:
            text: Text to speak
        """
        if not self.is_available:
            logger.warning("TTS not available")
            return
        
        if not text:
            return
        
        try:
            logger.debug(f"Speaking: '{text}'")
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            logger.error(f"Error speaking: {e}")
    
    def speak_async(self, text: str) -> None:
        """
        Speak text asynchronously (non-blocking).
        
        Uses a queue and background thread to avoid blocking the UI.
        
        Args:
            text: Text to speak
        """
        if not self.is_available:
            logger.warning("TTS not available")
            return
        
        if not text:
            return
        
        # Ensure worker thread is running
        self._ensure_worker_running()
        
        # Add to queue
        self._speech_queue.put(text)
        logger.debug(f"Queued for speaking: '{text}'")
    
    def _ensure_worker_running(self) -> None:
        """Ensure the background worker thread is running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_requested = False
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Background worker that processes speech queue."""
        # Initialize COM for this thread (required by pyttsx3/SAPI on Windows)
        _com_initialized = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            _com_initialized = True
        except (ImportError, Exception):
            pass

        # Create a new engine for this thread
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
        except Exception as e:
            logger.error(f"Failed to init engine in worker: {e}")
            return
        
        while not self._stop_requested:
            try:
                # Get text from queue with timeout
                text = self._speech_queue.get(timeout=0.5)
                
                if text is None:  # Poison pill
                    break
                
                self._is_speaking = True
                engine.say(text)
                engine.runAndWait()
                self._is_speaking = False
                
                self._speech_queue.task_done()
                
            except queue.Empty:
                continue
            except RuntimeError:
                # Engine may be in bad state, reinitialize
                logger.warning("TTS engine RuntimeError — reinitializing")
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.rate)
                    engine.setProperty('volume', self.volume)
                except Exception:
                    pass
                self._is_speaking = False
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self._is_speaking = False

        # Clean up COM
        if _com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
    
    def stop(self) -> None:
        """Stop current speech and clear queue."""
        if not self.is_available:
            return
        
        try:
            # Clear the queue
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Stop the engine
            self._engine.stop()
            
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the TTS engine and worker thread."""
        self._stop_requested = True
        
        # Add poison pill to queue
        self._speech_queue.put(None)
        
        # Wait for worker to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        
        logger.info("TTS engine shutdown complete")
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    def set_rate(self, rate: int) -> None:
        """
        Set speech rate.
        
        Args:
            rate: Words per minute (typically 100-200)
        """
        self.rate = rate
        if self._engine:
            self._engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float) -> None:
        """
        Set volume level.
        
        Args:
            volume: Volume from 0.0 to 1.0
        """
        self.volume = max(0.0, min(1.0, volume))
        if self._engine:
            self._engine.setProperty('volume', self.volume)
    
    def get_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice objects
        """
        if not self._engine:
            return []
        
        return self._engine.getProperty('voices')
    
    def set_voice(self, voice_id: str) -> None:
        """
        Set voice by ID.
        
        Args:
            voice_id: Voice identifier
        """
        if self._engine:
            self._engine.setProperty('voice', voice_id)
    
    def speak_letters(self, letters: list, delay_ms: int = 500) -> None:
        """
        Speak individual letters with delay between them.
        
        Useful for spelling out words shown as ISL signs.
        
        Args:
            letters: List of letters to speak
            delay_ms: Delay between letters in milliseconds
        """
        import time
        
        for letter in letters:
            self.speak(letter)
            time.sleep(delay_ms / 1000.0)


class MockTextToSpeech:
    """
    Mock TTS for testing when pyttsx3 is not available.
    """
    
    def __init__(self, *args, **kwargs):
        logger.info("Using mock text-to-speech")
    
    @property
    def is_available(self) -> bool:
        return True
    
    def speak(self, text: str) -> None:
        print(f"[MOCK TTS]: {text}")
    
    def speak_async(self, text: str) -> None:
        print(f"[MOCK TTS ASYNC]: {text}")
    
    def stop(self) -> None:
        pass
    
    def shutdown(self) -> None:
        pass
    
    @property
    def is_speaking(self) -> bool:
        return False


def create_tts(rate: int = 150, volume: float = 0.9, use_mock: bool = False) -> TextToSpeech:
    """
    Factory function to create a TTS engine.
    
    Args:
        rate: Speech rate
        volume: Volume level
        use_mock: If True, return mock TTS for testing
        
    Returns:
        TTS engine instance
    """
    if use_mock or not PYTTSX3_AVAILABLE:
        return MockTextToSpeech()
    
    return TextToSpeech(rate, volume)


if __name__ == "__main__":
    print("Testing TextToSpeech...")
    print(f"pyttsx3 available: {PYTTSX3_AVAILABLE}")
    
    if PYTTSX3_AVAILABLE:
        tts = TextToSpeech()
        
        if tts.is_available:
            print("\nTesting synchronous speech...")
            tts.speak("Hello, I am the ISL translation system.")
            
            print("\nTesting async speech...")
            tts.speak_async("This is spoken asynchronously.")
            tts.speak_async("Second message in queue.")
            
            import time
            time.sleep(5)  # Wait for async speech
            
            tts.shutdown()
            print("Test complete!")
        else:
            print("TTS not available")
    else:
        print("\npyttsx3 not installed.")
        print("Install with: pip install pyttsx3")
