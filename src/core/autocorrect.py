"""
Module: autocorrect.py
Description: Autocorrect and spelling suggestion for ISL-to-text
Author: Hackathon Team
Date: 2026
"""
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Try to import TextBlob for spell checking
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("TextBlob available for autocorrect")
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available - autocorrect disabled")


class AutoCorrect:
    """Autocorrect and spelling suggestion using NLP."""
    
    def __init__(self):
        """Initialize autocorrect."""
        self.enabled = TEXTBLOB_AVAILABLE
        self.min_word_length = 2  # Minimum word length to check
        
    def correct_word(self, word: str) -> str:
        """
        Correct a single word.
        
        Args:
            word: Word to correct
            
        Returns:
            Corrected word
        """
        if not self.enabled or not word or len(word) < self.min_word_length:
            return word
        
        try:
            # Preserve case
            is_upper = word.isupper()
            is_title = word.istitle()
            
            # Get correction
            blob = TextBlob(word.lower())
            corrected = str(blob.correct())
            
            # Restore case
            if is_upper:
                corrected = corrected.upper()
            elif is_title:
                corrected = corrected.capitalize()
            
            return corrected
        except Exception as e:
            logger.error(f"Error correcting word '{word}': {e}")
            return word
    
    def get_suggestions(self, word: str, max_suggestions: int = 3) -> List[str]:
        """
        Get spelling suggestions for a word.
        
        Args:
            word: Word to get suggestions for
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested corrections
        """
        if not self.enabled or not word or len(word) < self.min_word_length:
            return []
        
        try:
            blob = TextBlob(word.lower())
            suggestions = blob.spellcheck()
            
            # Filter and sort by confidence
            suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
            suggestions = [s[0] for s in suggestions[:max_suggestions]]
            
            # Preserve case of original word
            is_upper = word.isupper()
            is_title = word.istitle()
            
            if is_upper:
                suggestions = [s.upper() for s in suggestions]
            elif is_title:
                suggestions = [s.capitalize() for s in suggestions]
            
            return suggestions
        except Exception as e:
            logger.error(f"Error getting suggestions for '{word}': {e}")
            return []
    
    def correct_text(self, text: str) -> str:
        """
        Correct an entire text.
        
        Args:
            text: Text to correct
            
        Returns:
            Corrected text
        """
        if not self.enabled or not text:
            return text
        
        try:
            blob = TextBlob(text)
            return str(blob.correct())
        except Exception as e:
            logger.error(f"Error correcting text: {e}")
            return text
    
    def is_word_correct(self, word: str) -> bool:
        """
        Check if a word is spelled correctly.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is correct, False otherwise
        """
        if not self.enabled or not word or len(word) < self.min_word_length:
            return True
        
        try:
            blob = TextBlob(word.lower())
            suggestions = blob.spellcheck()
            
            # If the first suggestion is the word itself with confidence 1.0, it's correct
            if suggestions and suggestions[0][0] == word.lower() and suggestions[0][1] == 1.0:
                return True
            
            # Also check if correction equals original
            corrected = str(blob.correct())
            return corrected.lower() == word.lower()
        except Exception as e:
            logger.error(f"Error checking word '{word}': {e}")
            return True
    
    def get_correction_with_confidence(self, word: str) -> Tuple[str, float]:
        """
        Get correction with confidence score.
        
        Args:
            word: Word to correct
            
        Returns:
            Tuple of (corrected_word, confidence)
        """
        if not self.enabled or not word or len(word) < self.min_word_length:
            return (word, 1.0)
        
        try:
            blob = TextBlob(word.lower())
            suggestions = blob.spellcheck()
            
            if suggestions:
                corrected, confidence = suggestions[0]
                
                # Preserve case
                is_upper = word.isupper()
                is_title = word.istitle()
                
                if is_upper:
                    corrected = corrected.upper()
                elif is_title:
                    corrected = corrected.capitalize()
                
                return (corrected, confidence)
            
            return (word, 1.0)
        except Exception as e:
            logger.error(f"Error getting correction with confidence for '{word}': {e}")
            return (word, 1.0)


# Global instance
autocorrect = AutoCorrect()
