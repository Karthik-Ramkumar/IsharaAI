"""
Module: speech_recognition.py
Description: Speech-to-text using Vosk with sounddevice
Author: Hackathon Team
Date: 2026

Uses sounddevice for audio capture (no PyAudio needed).
Falls back to soundcard if sounddevice fails.
"""
import os
import sys
import json
import logging
import wave
import queue
import threading
import numpy as np
from typing import Optional, Callable

# Setup logging
logger = logging.getLogger(__name__)

# Audio libraries availability
SOUNDDEVICE_AVAILABLE = False
SOUNDCARD_AVAILABLE = False
VOSK_AVAILABLE = False
SR_AVAILABLE = False

# Try to import sounddevice
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    logger.info("sounddevice available")
except ImportError:
    logger.warning("sounddevice not installed")

# Try to import soundcard as fallback
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
    logger.info("soundcard available")
except ImportError:
    logger.warning("soundcard not installed")

# Try to import Vosk
try:
    import vosk
    VOSK_AVAILABLE = True
    logger.info("Vosk available")
except ImportError:
    logger.warning("Vosk not installed")

# Try SpeechRecognition as last resort
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    pass


class VoskRecognizerWithSounddevice:
    """
    Vosk speech recognition using sounddevice for audio capture.
    """
    
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SIZE = 8000  # ~0.5 seconds of audio
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self._is_initialized = False
        self._audio_queue = queue.Queue()
        
        self._initialize()
    
    def _initialize(self) -> bool:
        if not VOSK_AVAILABLE:
            logger.error("Vosk not available")
            return False
        
        if not SOUNDDEVICE_AVAILABLE and not SOUNDCARD_AVAILABLE:
            logger.error("No audio library available")
            return False
        
        # Get model path
        if not self.model_path:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            try:
                import config
                self.model_path = config.VOSK_MODEL_DIR
            except ImportError:
                self.model_path = "models/vosk-model-small-en-us-0.15"
        
        if not os.path.exists(self.model_path):
            logger.error(f"Vosk model not found: {self.model_path}")
            return False
        
        try:
            vosk.SetLogLevel(-1)
            logger.info(f"Loading Vosk model from: {self.model_path}")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
            self._is_initialized = True
            logger.info("Vosk recognizer initialized with sounddevice")
            return True
        except Exception as e:
            logger.error(f"Failed to init Vosk: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        return self._is_initialized and (SOUNDDEVICE_AVAILABLE or SOUNDCARD_AVAILABLE)
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        self._audio_queue.put(bytes(indata))
    
    def listen_continuously(self, callback: Callable, stop_event: threading.Event):
        """
        Record audio continuously until stop_event is set.
        callback(text: str, is_final: bool)
        """
        if not self.is_available:
            return

        # Reset recognizer
        self.recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
        
        try:
            # Use RawInputStream for better control
            with sd.RawInputStream(samplerate=self.SAMPLE_RATE, blocksize=4000,
                                   dtype='int16', channels=1) as stream:
                logger.info("Listening continuously...")
                
                while not stop_event.is_set():
                    data, overflowed = stream.read(4000)
                    if overflowed:
                        logger.warning("Audio buffer overflow")
                    
                    if self.recognizer.AcceptWaveform(bytes(data)):
                        result = json.loads(self.recognizer.Result())
                        if result.get('text'):
                            callback(result['text'], False)
        
        except Exception as e:
            logger.error(f"Continuous listening error: {e}")
        
        # Final result
        final = json.loads(self.recognizer.FinalResult())
        if final.get('text'):
            callback(final['text'], True)

    def recognize_from_microphone(self, duration: int = 5,
                                  callback: Optional[Callable] = None) -> str:
        """Record and recognize speech."""
        if not self.is_available:
            logger.error("Recognizer not available")
            return ""
        
        # Reset recognizer
        self.recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"Recording for {duration} seconds...")
        
        if SOUNDDEVICE_AVAILABLE:
            return self._record_with_sounddevice(duration, callback)
        elif SOUNDCARD_AVAILABLE:
            return self._record_with_soundcard(duration, callback)
        
        return ""
    
    def _record_with_sounddevice(self, duration: int, callback: Optional[Callable]) -> str:
        """Record using sounddevice."""
        try:
            # Record audio
            logger.info("Recording with sounddevice...")
            recording = sd.rec(
                int(duration * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='int16'
            )
            sd.wait()
            
            # Convert to bytes and process
            audio_bytes = recording.tobytes()
            
            # Process in chunks
            chunk_size = 4000
            text_parts = []
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                if self.recognizer.AcceptWaveform(chunk):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        text_parts.append(result['text'])
                        if callback:
                            callback(result['text'])
            
            # Get final result
            final = json.loads(self.recognizer.FinalResult())
            if final.get('text'):
                text_parts.append(final['text'])
            
            text = ' '.join(text_parts).strip()
            logger.info(f"Recognized: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"sounddevice error: {e}")
            # Try fallback
            if SOUNDCARD_AVAILABLE:
                return self._record_with_soundcard(duration, callback)
            return ""
    
    def _record_with_soundcard(self, duration: int, callback: Optional[Callable]) -> str:
        """Record using soundcard as fallback."""
        try:
            logger.info("Recording with soundcard...")
            
            mic = sc.default_microphone()
            
            with mic.recorder(samplerate=self.SAMPLE_RATE, channels=1) as recorder:
                # Record
                data = recorder.record(numframes=int(duration * self.SAMPLE_RATE))
                
                # Convert to int16
                audio_data = (data * 32767).astype(np.int16)
                audio_bytes = audio_data.tobytes()
                
                # Process
                chunk_size = 4000
                text_parts = []
                
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i+chunk_size]
                    if self.recognizer.AcceptWaveform(chunk):
                        result = json.loads(self.recognizer.Result())
                        if result.get('text'):
                            text_parts.append(result['text'])
                
                final = json.loads(self.recognizer.FinalResult())
                if final.get('text'):
                    text_parts.append(final['text'])
                
                text = ' '.join(text_parts).strip()
                logger.info(f"Recognized: '{text}'")
                return text
                
        except Exception as e:
            logger.error(f"soundcard error: {e}")
            return ""


class GoogleSpeechRecognizer:
    """Fallback using Google Speech Recognition."""
    
    def __init__(self):
        self.recognizer = None
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
    
    @property
    def is_available(self) -> bool:
        return SR_AVAILABLE and self.recognizer is not None
    
    def recognize_from_microphone(self, duration: int = 5,
                                  callback: Optional[Callable] = None) -> str:
        if not self.is_available:
            return ""
        
        try:
            with sr.Microphone() as source:
                logger.info("Listening with Google Speech...")
                if callback:
                    callback("Listening...")
                
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
                
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized: '{text}'")
                return text
                
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            logger.error(f"Google speech error: {e}")
            return ""


class MockRecognizer:
    """Mock for testing."""
    
    @property
    def is_available(self) -> bool:
        return True
    
    def recognize_from_microphone(self, duration: int = 5,
                                  callback: Optional[Callable] = None) -> str:
        return "hello world"


def create_recognizer(model_path: Optional[str] = None, use_mock: bool = False):
    """
    Create the best available speech recognizer.
    
    Priority: Vosk+sounddevice > Vosk+soundcard > Google > Mock
    """
    if use_mock:
        return MockRecognizer()
    
    # Try Vosk with sounddevice/soundcard
    if VOSK_AVAILABLE and (SOUNDDEVICE_AVAILABLE or SOUNDCARD_AVAILABLE):
        rec = VoskRecognizerWithSounddevice(model_path)
        if rec.is_available:
            logger.info("Using Vosk with sounddevice/soundcard")
            return rec
    
    # Try Google
    if SR_AVAILABLE:
        rec = GoogleSpeechRecognizer()
        if rec.is_available:
            logger.info("Using Google Speech Recognition")
            return rec
    
    # Mock fallback
    logger.warning("No recognizer available - using mock")
    return MockRecognizer()


def test_audio():
    """Test audio recording."""
    print("=" * 50)
    print("Testing Audio Recording")
    print("=" * 50)
    print(f"sounddevice: {SOUNDDEVICE_AVAILABLE}")
    print(f"soundcard: {SOUNDCARD_AVAILABLE}")
    print(f"Vosk: {VOSK_AVAILABLE}")
    
    if SOUNDDEVICE_AVAILABLE:
        print("\nTesting sounddevice...")
        try:
            devices = sd.query_devices()
            print(f"Default input device: {sd.query_devices(kind='input')['name']}")
            
            print("Recording 2 seconds...")
            audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
            print(f"Recorded {len(audio)} samples")
            print("✓ sounddevice working!")
        except Exception as e:
            print(f"✗ sounddevice error: {e}")
    
    if VOSK_AVAILABLE:
        print(f"\nVosk model path check...")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        try:
            import config
            model_path = config.VOSK_MODEL_DIR
            exists = os.path.exists(model_path)
            print(f"Model path: {model_path}")
            print(f"Model exists: {exists}")
        except Exception as e:
            print(f"Config error: {e}")


if __name__ == "__main__":
    test_audio()
    
    print("\n" + "=" * 50)
    print("Testing Speech Recognition")
    print("=" * 50)
    
    recognizer = create_recognizer()
    print(f"Recognizer: {type(recognizer).__name__}")
    print(f"Available: {recognizer.is_available}")
    
    if recognizer.is_available and not isinstance(recognizer, MockRecognizer):
        print("\nSpeak now (5 seconds)...")
        text = recognizer.recognize_from_microphone(duration=5)
        print(f"Result: '{text}'")
