"""
Module: speech_to_isl.py
Description: Pipeline 1: Speech → Text → ISL Images
Author: Hackathon Team
Date: 2026

This pipeline converts spoken audio into ISL sign language images.
It combines speech recognition with the text-to-ISL pipeline.
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Callable

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.core.speech_recognition import VoskRecognizerWithSounddevice, create_recognizer
from src.pipelines.text_to_isl import TextToISLPipeline

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class SpeechToISLPipeline:
    """
    Pipeline for converting speech to ISL image sequences.
    
    This pipeline:
    1. Records audio from microphone (or accepts audio input)
    2. Converts speech to text using Vosk
    3. Translates text to ISL images using TextToISLPipeline
    
    Example usage:
        pipeline = SpeechToISLPipeline()
        if pipeline.is_ready():
            result = pipeline.translate_from_speech(duration=5)
            print(f"Recognized: {result['text']}")
            for item in result['signs']:
                print(f"Sign: {item['char']}")
    """
    
    def __init__(self, vosk_model_path: Optional[str] = None, lazy_load: bool = False):
        """
        Initialize the Speech to ISL pipeline.
        
        Args:
            vosk_model_path: Path to Vosk model directory
            lazy_load: If True, delay loading until first use
        """
        self.vosk_model_path = vosk_model_path or config.VOSK_MODEL_DIR
        
        # Initialize components
        self.speech_recognizer = None
        self.text_to_isl = None
        
        self._is_loaded = False
        
        if not lazy_load:
            self._load_components()
        
        logger.info("SpeechToISLPipeline initialized")
    
    def _load_components(self) -> None:
        """Load all pipeline components."""
        if self._is_loaded:
            return
        
        # Initialize speech recognizer
        self.speech_recognizer = create_recognizer(self.vosk_model_path)
        
        # Initialize text-to-ISL pipeline
        self.text_to_isl = TextToISLPipeline()
        
        self._is_loaded = True
        logger.info("Pipeline components loaded")
    
    def ensure_loaded(self) -> None:
        """Ensure all components are loaded."""
        if not self._is_loaded:
            self._load_components()
    
    def is_ready(self) -> bool:
        """
        Check if the pipeline is ready for speech recognition.
        
        Returns:
            True if all components are ready
        """
        self.ensure_loaded()
        
        speech_ready = self.speech_recognizer and self.speech_recognizer.is_available
        text_ready = self.text_to_isl and self.text_to_isl.is_ready
        
        return speech_ready and text_ready
    
    def translate_from_speech(self, duration: int = 5,
                              on_partial: Optional[Callable[[str], None]] = None) -> Dict:
        """
        Record speech and convert to ISL images.
        
        Args:
            duration: Recording duration in seconds
            on_partial: Optional callback for partial recognition results
            
        Returns:
            Dictionary containing:
                - 'text': Recognized text
                - 'normalized_text': Cleaned text
                - 'signs': List of ISL sign dictionaries (same as text_to_isl)
                - 'word_count': Number of words recognized
                - 'sign_count': Number of signs generated
                - 'success': Whether recognition was successful
        """
        self.ensure_loaded()
        
        result = {
            'text': '',
            'normalized_text': '',
            'signs': [],
            'word_count': 0,
            'sign_count': 0,
            'success': False
        }
        
        if not self.is_ready():
            logger.error("Pipeline not ready for speech recognition")
            return result
        
        logger.info(f"Starting speech recognition for {duration} seconds...")
        
        # Step 1: Recognize speech
        recognized_text = self.speech_recognizer.recognize_from_microphone(
            duration=duration,
            callback=on_partial
        )
        
        if not recognized_text:
            logger.warning("No speech recognized")
            return result
        
        result['text'] = recognized_text
        result['word_count'] = len(recognized_text.split())
        
        # Step 2: Translate to ISL
        signs = self.text_to_isl.translate(recognized_text)
        
        result['signs'] = signs
        result['sign_count'] = len(signs)
        result['normalized_text'] = self.text_to_isl.translate_to_display_string(recognized_text)
        result['success'] = True
        
        logger.info(f"Speech translated: {result['word_count']} words -> {result['sign_count']} signs")
        
        return result
    
    def translate_text_to_isl(self, text: str) -> List[Dict]:
        """
        Direct text-to-ISL translation (bypasses speech recognition).
        
        Useful for testing or when text is already available.
        
        Args:
            text: Text to translate
            
        Returns:
            List of ISL sign dictionaries
        """
        self.ensure_loaded()
        return self.text_to_isl.translate(text)
    
    def start_continuous_recognition(self, 
                                     on_result: Callable[[Dict], None],
                                     on_partial: Optional[Callable[[str], None]] = None) -> None:
        """
        Start continuous speech recognition.
        
        Results are delivered via callback as speech is recognized.
        
        Args:
            on_result: Callback for final results (receives dict with text and signs)
            on_partial: Optional callback for partial text
        """
        self.ensure_loaded()
        
        if not self.is_ready():
            logger.error("Pipeline not ready")
            return
        
        def handle_result(text: str, is_final: bool):
            if is_final and text:
                signs = self.text_to_isl.translate(text)
                on_result({
                    'text': text,
                    'signs': signs,
                    'is_final': True
                })
            elif on_partial:
                on_partial(text)
        
        self.speech_recognizer.start_streaming(handle_result)
    
    def stop_continuous_recognition(self) -> None:
        """Stop continuous speech recognition."""
        if self.speech_recognizer:
            self.speech_recognizer.stop_streaming()
    
    def get_supported_signs(self) -> List[str]:
        """Get list of supported ISL signs."""
        self.ensure_loaded()
        return self.text_to_isl.get_supported_signs()
    
    def get_status(self) -> Dict:
        """
        Get pipeline status information.
        
        Returns:
            Dictionary with status details
        """
        self.ensure_loaded()
        
        return {
            'speech_available': self.speech_recognizer.is_available if self.speech_recognizer else False,
            'text_to_isl_ready': self.text_to_isl.is_ready if self.text_to_isl else False,
            'model_path': self.vosk_model_path,
            'model_exists': os.path.exists(self.vosk_model_path),
            'supported_signs': len(self.get_supported_signs())
        }


def create_pipeline(vosk_model_path: Optional[str] = None) -> SpeechToISLPipeline:
    """
    Factory function to create a SpeechToISLPipeline.
    
    Args:
        vosk_model_path: Path to Vosk model
        
    Returns:
        Initialized SpeechToISLPipeline
    """
    return SpeechToISLPipeline(vosk_model_path)


# Quick test
def test_pipeline():
    """Run basic tests on the pipeline."""
    print("Testing SpeechToISLPipeline...")
    
    try:
        pipeline = SpeechToISLPipeline()
        
        # Check status
        status = pipeline.get_status()
        print(f"\nPipeline Status:")
        print(f"  Speech available: {status['speech_available']}")
        print(f"  Text-to-ISL ready: {status['text_to_isl_ready']}")
        print(f"  Model exists: {status['model_exists']}")
        print(f"  Supported signs: {status['supported_signs']}")
        
        if not status['speech_available']:
            print("\n⚠ Speech recognition not available.")
            print("  Please download a Vosk model and place it at:")
            print(f"  {status['model_path']}")
            print("\n  Download from: https://alphacephei.com/vosk/models")
            print("  Recommended: vosk-model-small-en-us-0.15 (40MB)")
            
            # Test just the text-to-ISL part
            print("\nTesting text-to-ISL only...")
            signs = pipeline.translate_text_to_isl("Hello World")
            if signs:
                print(f"  'Hello World' -> {len(signs)} signs")
                print(f"  Signs: {[s['char'] for s in signs]}")
            return True
        
        # Test with speech
        print("\nSpeak now (5 seconds)...")
        result = pipeline.translate_from_speech(duration=5)
        
        if result['success']:
            print(f"\nRecognized: '{result['text']}'")
            print(f"Signs: {[s['char'] for s in result['signs']]}")
            print(f"Total signs: {result['sign_count']}")
        else:
            print("No speech recognized")
        
        print("\n✓ Test complete!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pipeline()
