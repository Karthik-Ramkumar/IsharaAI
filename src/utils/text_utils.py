"""
Module: text_utils.py
Description: Text normalization, tokenization, and character matching for ISL translation
Author: Hackathon Team
Date: 2026
"""
import re
import logging
from typing import List, Tuple

# Setup logging
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize input text for ISL translation.
    
    Steps:
    1. Convert to lowercase
    2. Remove extra whitespace
    3. Keep only alphanumeric characters and spaces
    
    Args:
        text: Raw input text
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalize_text("Hello World! 123")
        'hello world 123'
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    logger.debug(f"Normalized text: '{text}'")
    return text


def tokenize_to_characters(text: str) -> List[str]:
    """
    Tokenize text into individual characters for ISL fingerspelling.
    
    This is the primary tokenization for ISL since the dataset contains
    individual letter and number signs (A-Z, 1-9).
    
    Args:
        text: Normalized text string
        
    Returns:
        List of individual characters (excluding spaces for display purposes)
        
    Example:
        >>> tokenize_to_characters("hello")
        ['h', 'e', 'l', 'l', 'o']
    """
    if not text:
        return []
    
    # Get individual characters, keeping spaces as word separators
    characters = list(text)
    
    logger.debug(f"Tokenized to {len(characters)} characters")
    return characters


def tokenize_to_words(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Normalized text string
        
    Returns:
        List of words
        
    Example:
        >>> tokenize_to_words("hello world")
        ['hello', 'world']
    """
    if not text:
        return []
    
    words = text.split()
    
    # Remove common filler words
    filler_words = {'um', 'uh', 'like', 'you know', 'actually', 'basically'}
    words = [w for w in words if w not in filler_words]
    
    logger.debug(f"Tokenized to {len(words)} words")
    return words


def filter_supported_characters(characters: List[str], supported_chars: set) -> Tuple[List[str], List[str]]:
    """
    Filter characters to only include those supported by the ISL dataset.
    
    Args:
        characters: List of characters to filter
        supported_chars: Set of supported characters
        
    Returns:
        Tuple of (supported_characters, unsupported_characters)
        
    Example:
        >>> filter_supported_characters(['h', 'e', 'l', 'l', 'o', ' '], {'h', 'e', 'l', 'o'})
        (['h', 'e', 'l', 'l', 'o'], [' '])
    """
    supported = []
    unsupported = []
    
    for char in characters:
        if char in supported_chars:
            supported.append(char)
        else:
            unsupported.append(char)
    
    if unsupported:
        unique_unsupported = list(set(unsupported))
        logger.warning(f"Unsupported characters: {unique_unsupported}")
    
    return supported, unsupported


def text_to_isl_sequence(text: str, supported_chars: set) -> Tuple[List[str], str]:
    """
    Convert input text to a sequence of ISL signs.
    
    This is the main function that combines normalization, tokenization,
    and filtering into a single pipeline.
    
    Args:
        text: Raw input text
        supported_chars: Set of supported characters from config
        
    Returns:
        Tuple of (list of characters for ISL display, cleaned text)
        
    Example:
        >>> text_to_isl_sequence("Hello!", {'h', 'e', 'l', 'o'})
        (['h', 'e', 'l', 'l', 'o'], 'hello')
    """
    # Step 1: Normalize
    normalized = normalize_text(text)
    
    # Step 2: Tokenize to characters
    characters = tokenize_to_characters(normalized)
    
    # Step 3: Filter supported characters (spaces become pauses in display)
    supported, _ = filter_supported_characters(characters, supported_chars)
    
    return supported, normalized


def spell_out_word(word: str, supported_chars: set) -> List[str]:
    """
    Spell out a word as individual ISL signs.
    
    Args:
        word: Word to spell out
        supported_chars: Set of supported characters
        
    Returns:
        List of characters that have corresponding ISL signs
    """
    characters = list(word.lower())
    supported, _ = filter_supported_characters(characters, supported_chars)
    return supported


def number_to_signs(number: str) -> List[str]:
    """
    Convert a number string to individual digit signs.
    
    Args:
        number: String representation of a number
        
    Returns:
        List of digit characters
        
    Example:
        >>> number_to_signs("123")
        ['1', '2', '3']
    """
    return [char for char in number if char.isdigit()]


def prepare_display_text(text: str) -> str:
    """
    Prepare text for display in the UI.
    
    Capitalizes appropriately and formats for readability.
    
    Args:
        text: Normalized text
        
    Returns:
        Formatted display text
    """
    if not text:
        return ""
    
    # Capitalize first letter of each sentence
    sentences = text.split('. ')
    sentences = [s.capitalize() if s else s for s in sentences]
    return '. '.join(sentences)


# Convenience function for quick testing
def process_input(text: str) -> dict:
    """
    Process input text and return analysis.
    
    Args:
        text: Raw input text
        
    Returns:
        Dictionary with processing results
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, '..')
    try:
        import config
        supported_chars = set(config.SUPPORTED_SIGNS.keys())
    except ImportError:
        # Fallback for testing
        supported_chars = set('abcdefghijklmnopqrstuvwxyz123456789')
    
    normalized = normalize_text(text)
    characters = tokenize_to_characters(normalized)
    words = tokenize_to_words(normalized)
    supported, unsupported = filter_supported_characters(characters, supported_chars)
    
    return {
        'original': text,
        'normalized': normalized,
        'characters': characters,
        'words': words,
        'supported_chars': supported,
        'unsupported_chars': unsupported,
        'char_count': len(characters),
        'supported_count': len(supported),
        'word_count': len(words)
    }


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Hello World!",
        "HELP ME 123",
        "Testing... 1, 2, 3!",
        "",
        "   Multiple   Spaces   ",
    ]
    
    for text in test_texts:
        print(f"\nInput: '{text}'")
        result = process_input(text)
        print(f"  Normalized: '{result['normalized']}'")
        print(f"  Characters: {result['supported_chars']}")
        print(f"  Words: {result['words']}")
