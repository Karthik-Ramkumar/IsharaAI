"""
Module: text_to_isl.py
Description: Pipeline 2: Text → ISL Images
Author: Hackathon Team
Date: 2026

This pipeline converts typed text into a sequence of ISL (Indian Sign Language)
images for display. It handles fingerspelling where each character maps to a sign.
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.text_utils import (
    normalize_text, 
    tokenize_to_characters,
    filter_supported_characters,
    text_to_isl_sequence
)
from src.utils.image_utils import ISLImageCache

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TextToISLPipeline:
    """
    Pipeline for converting text to ISL image sequences.
    
    This pipeline:
    1. Normalizes input text (lowercase, remove punctuation)
    2. Tokenizes into individual characters
    3. Filters to supported ISL signs (A-Z, 1-9)
    4. Returns corresponding ISL images
    
    Example usage:
        pipeline = TextToISLPipeline()
        result = pipeline.translate("Hello")
        for item in result:
            print(f"Char: {item['char']}, Has Image: {item['image'] is not None}")
    """
    
    def __init__(self, lazy_load: bool = False):
        """
        Initialize the Text to ISL pipeline.
        
        Args:
            lazy_load: If True, don't load images until first use
        """
        self.supported_chars = set(config.SUPPORTED_SIGNS.keys())
        self.sign_mapping = config.SUPPORTED_SIGNS
        
        # Initialize image cache
        self.image_cache = ISLImageCache(
            image_dir=config.RAW_DATA_DIR,
            display_size=config.DISPLAY_SIZE
        )
        
        self._is_loaded = False
        
        if not lazy_load:
            self._load_images()
        
        logger.info("TextToISLPipeline initialized")
    
    def _load_images(self) -> None:
        """Load all ISL images into cache."""
        if self._is_loaded:
            return
        
        count = self.image_cache.load_all_images(self.sign_mapping)
        self._is_loaded = True
        logger.info(f"Loaded {count} ISL sign images")
    
    def ensure_loaded(self) -> None:
        """Ensure images are loaded (for lazy loading)."""
        if not self._is_loaded:
            self._load_images()
    
    def translate(self, text: str) -> List[Dict]:
        """
        Convert text to ISL image sequence.
        
        Args:
            text: Input text to translate
            
        Returns:
            List of dictionaries, each containing:
                - 'char': The character being represented
                - 'image': PIL Image of the ISL sign
                - 'confidence': 1.0 for direct text translation
                - 'found': Whether the sign was found in the dataset
                - 'folder': The dataset folder name for this sign
                
        Example:
            >>> pipeline = TextToISLPipeline()
            >>> result = pipeline.translate("Hi")
            >>> len(result)
            2
            >>> result[0]['char']
            'h'
        """
        self.ensure_loaded()
        
        if not text:
            logger.debug("Empty input received")
            return []
        
        # Step 1: Normalize and tokenize
        supported_chars, normalized = text_to_isl_sequence(text, self.supported_chars)
        
        if not supported_chars:
            logger.warning(f"No supported characters in input: '{text}'")
            return []
        
        logger.info(f"Translating: '{normalized}' -> {len(supported_chars)} signs")
        
        # Step 2: Get images for each character
        result = []
        for char in supported_chars:
            image = self.image_cache.get_image(char)
            folder_name = self.sign_mapping.get(char, char.upper())
            
            result.append({
                'char': char,
                'image': image if image else self.image_cache.get_placeholder(),
                'confidence': 1.0,  # Text translation is deterministic
                'found': image is not None,
                'folder': folder_name
            })
        
        found_count = sum(1 for r in result if r['found'])
        logger.info(f"Translation complete: {found_count}/{len(result)} signs found")
        
        return result
    
    def translate_with_spaces(self, text: str) -> List[Dict]:
        """
        Convert text to ISL sequence, preserving word boundaries.
        
        Includes 'space' markers between words for display purposes.
        
        Args:
            text: Input text to translate
            
        Returns:
            List of dictionaries, including space markers
        """
        self.ensure_loaded()
        
        if not text:
            return []
        
        normalized = normalize_text(text)
        words = normalized.split()
        
        result = []
        for i, word in enumerate(words):
            # Translate the word
            word_result = self.translate(word)
            result.extend(word_result)
            
            # Add space marker between words (not after the last word)
            if i < len(words) - 1:
                result.append({
                    'char': ' ',
                    'image': None,
                    'confidence': 1.0,
                    'found': True,
                    'folder': None,
                    'is_space': True
                })
        
        return result
    
    def get_supported_signs(self) -> List[str]:
        """
        Get list of all supported signs.
        
        Returns:
            List of supported characters
        """
        return list(self.supported_chars)
    
    def get_available_signs(self) -> List[str]:
        """
        Get list of signs that have images loaded.
        
        Returns:
            List of characters with loaded images
        """
        self.ensure_loaded()
        return self.image_cache.available_signs
    
    def is_supported(self, char: str) -> bool:
        """
        Check if a character is supported.
        
        Args:
            char: Character to check
            
        Returns:
            True if supported, False otherwise
        """
        return char.lower() in self.supported_chars
    
    def get_translation_stats(self, text: str) -> Dict:
        """
        Get statistics about text translation without actually translating.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with translation statistics
        """
        normalized = normalize_text(text)
        characters = tokenize_to_characters(normalized)
        supported, unsupported = filter_supported_characters(characters, self.supported_chars)
        
        return {
            'original_text': text,
            'normalized_text': normalized,
            'total_characters': len(characters),
            'supported_characters': len(supported),
            'unsupported_characters': len(unsupported),
            'support_rate': len(supported) / len(characters) if characters else 0,
            'unsupported_list': list(set(unsupported))
        }
    
    def translate_to_display_string(self, text: str) -> str:
        """
        Get the string that will be displayed as ISL signs.
        
        Useful for preview purposes.
        
        Args:
            text: Input text
            
        Returns:
            String of characters that will be translated
        """
        supported, _ = text_to_isl_sequence(text, self.supported_chars)
        return ''.join(supported)
    
    @property
    def is_ready(self) -> bool:
        """Check if the pipeline is ready (images loaded)."""
        return self._is_loaded and self.image_cache.loaded_count > 0


def create_pipeline() -> TextToISLPipeline:
    """
    Factory function to create a TextToISLPipeline.
    
    Returns:
        Initialized TextToISLPipeline
    """
    return TextToISLPipeline()


# Quick test
def test_pipeline():
    """Run basic tests on the pipeline."""
    print("Testing TextToISLPipeline...")
    
    try:
        pipeline = TextToISLPipeline()
        
        # Test cases
        test_cases = [
            "Hello",
            "WORLD",
            "Test 123",
            "Hello World!",
            "",
            "   spaces   ",
            "special@#$chars"
        ]
        
        for text in test_cases:
            print(f"\nInput: '{text}'")
            
            # Get stats first
            stats = pipeline.get_translation_stats(text)
            print(f"  Stats: {stats['supported_characters']} supported, "
                  f"{stats['unsupported_characters']} unsupported")
            
            # Translate
            result = pipeline.translate(text)
            if result:
                chars = [r['char'] for r in result]
                found = sum(1 for r in result if r['found'])
                print(f"  Result: {''.join(chars)} ({found}/{len(result)} images found)")
            else:
                print("  Result: (empty)")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pipeline()
