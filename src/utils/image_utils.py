"""
Module: image_utils.py
Description: Image loading, caching, and optimization for ISL display
Author: Hackathon Team
Date: 2026
"""
import os
import logging
import shutil
from typing import Dict, Optional, List, Tuple
from PIL import Image
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


class ISLImageCache:
    """
    Manages loading and caching of ISL sign images.
    
    Pre-loads all images at startup for instant access during translation.
    """
    
    def __init__(self, image_dir: str, display_size: Tuple[int, int] = (150, 150)):
        """
        Initialize the image cache.
        
        Args:
            image_dir: Directory containing ISL images (organized by sign folders)
            display_size: Size to resize images for display
        """
        self.image_dir = image_dir
        self.display_size = display_size
        self._cache: Dict[str, Image.Image] = {}
        self._placeholder: Optional[Image.Image] = None
        
        logger.info(f"Initializing image cache from: {image_dir}")
    
    def load_all_images(self, sign_mapping: Dict[str, str]) -> int:
        """
        Pre-load all ISL images into memory.
        
        Args:
            sign_mapping: Dictionary mapping characters to folder names
                         e.g., {'a': 'A', 'b': 'B', ...}
        
        Returns:
            Number of successfully loaded images
        """
        loaded_count = 0
        
        for char, folder_name in sign_mapping.items():
            folder_path = os.path.join(self.image_dir, folder_name)
            
            if not os.path.exists(folder_path):
                logger.warning(f"Folder not found for '{char}': {folder_path}")
                continue
            
            # Get the first image in the folder as representative
            image = self._load_representative_image(folder_path)
            
            if image:
                self._cache[char] = image
                loaded_count += 1
                logger.debug(f"Loaded image for '{char}'")
            else:
                logger.warning(f"Could not load image for '{char}'")
        
        logger.info(f"Loaded {loaded_count} images into cache")
        return loaded_count
    
    def _load_representative_image(self, folder_path: str) -> Optional[Image.Image]:
        """
        Load a representative image from a folder.
        
        Selects a middle image (not first/last) for better representation.
        
        Args:
            folder_path: Path to the sign folder
            
        Returns:
            PIL Image or None
        """
        try:
            # Get all image files
            image_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            
            if not image_files:
                return None
            
            # Select middle image for better representation
            middle_idx = len(image_files) // 2
            image_path = os.path.join(folder_path, image_files[middle_idx])
            
            return self._load_and_resize(image_path)
            
        except Exception as e:
            logger.error(f"Error loading from {folder_path}: {e}")
            return None
    
    def _load_and_resize(self, image_path: str) -> Optional[Image.Image]:
        """
        Load an image and resize for display.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Resized PIL Image or None
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize while maintaining aspect ratio
            image.thumbnail(self.display_size, Image.Resampling.LANCZOS)
            
            # Create a new image with exact display size (centered)
            result = Image.new('RGB', self.display_size, (255, 255, 255))
            paste_x = (self.display_size[0] - image.width) // 2
            paste_y = (self.display_size[1] - image.height) // 2
            result.paste(image, (paste_x, paste_y))
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def get_image(self, char: str) -> Optional[Image.Image]:
        """
        Get the cached image for a character.
        
        Args:
            char: Character to get image for
            
        Returns:
            PIL Image or None if not found
        """
        return self._cache.get(char.lower())
    
    def get_placeholder(self) -> Image.Image:
        """
        Get a placeholder image for unsupported characters.
        
        Returns:
            PIL Image placeholder
        """
        if self._placeholder is None:
            # Create a simple placeholder
            self._placeholder = Image.new('RGB', self.display_size, (200, 200, 200))
            # Could add text overlay here if needed
        
        return self._placeholder
    
    def get_images_for_sequence(self, characters: List[str]) -> List[Dict]:
        """
        Get images for a sequence of characters.
        
        Args:
            characters: List of characters
            
        Returns:
            List of dictionaries with 'char', 'image', and 'found' keys
        """
        result = []
        
        for char in characters:
            image = self.get_image(char)
            result.append({
                'char': char,
                'image': image if image else self.get_placeholder(),
                'found': image is not None
            })
        
        return result
    
    @property
    def loaded_count(self) -> int:
        """Number of images currently in cache."""
        return len(self._cache)
    
    @property
    def available_signs(self) -> List[str]:
        """List of signs available in cache."""
        return list(self._cache.keys())


def prepare_display_images(raw_data_dir: str, output_dir: str, 
                          sign_mapping: Dict[str, str],
                          display_size: Tuple[int, int] = (150, 150)) -> int:
    """
    Copy and optimize representative images for UI display.
    
    This creates a processed folder with one optimized image per sign.
    
    Args:
        raw_data_dir: Source directory with raw ISL images
        output_dir: Destination directory for processed images
        sign_mapping: Dictionary mapping characters to folder names
        display_size: Size to resize images to
        
    Returns:
        Number of images processed
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    
    logger.info(f"Preparing display images from {raw_data_dir} to {output_dir}")
    
    for char, folder_name in sign_mapping.items():
        source_folder = os.path.join(raw_data_dir, folder_name)
        
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder not found: {source_folder}")
            continue
        
        # Create output folder for this sign
        sign_output_dir = os.path.join(output_dir, char)
        os.makedirs(sign_output_dir, exist_ok=True)
        
        # Get representative image
        try:
            image_files = sorted([
                f for f in os.listdir(source_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            
            if not image_files:
                continue
            
            # Select multiple images for variety (first, middle, last)
            indices = [0, len(image_files) // 2, -1]
            
            for i, idx in enumerate(indices):
                src_path = os.path.join(source_folder, image_files[idx])
                dst_path = os.path.join(sign_output_dir, f"{char}_{i}.jpg")
                
                # Load, resize, and save
                image = Image.open(src_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                image.save(dst_path, 'JPEG', quality=90)
            
            processed_count += 1
            logger.debug(f"Processed images for '{char}'")
            
        except Exception as e:
            logger.error(f"Error processing {folder_name}: {e}")
    
    logger.info(f"Processed {processed_count} signs for display")
    return processed_count


def load_single_image(image_path: str, size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """
    Load a single image from disk.
    
    Args:
        image_path: Path to the image
        size: Optional size to resize to
        
    Returns:
        PIL Image or None
    """
    try:
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if size:
            image.thumbnail(size, Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading {image_path}: {e}")
        return None


def get_random_image_from_folder(folder_path: str, size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """
    Get a random image from a folder.
    
    Args:
        folder_path: Path to folder containing images
        size: Optional size to resize to
        
    Returns:
        PIL Image or None
    """
    import random
    
    try:
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        if not image_files:
            return None
        
        selected = random.choice(image_files)
        return load_single_image(os.path.join(folder_path, selected), size)
        
    except Exception as e:
        logger.error(f"Error getting random image from {folder_path}: {e}")
        return None


def create_image_grid(images: List[Image.Image], 
                     columns: int = 4, 
                     padding: int = 10) -> Image.Image:
    """
    Create a grid of images for display.
    
    Args:
        images: List of PIL Images
        columns: Number of columns in the grid
        padding: Padding between images
        
    Returns:
        Combined PIL Image
    """
    if not images:
        return Image.new('RGB', (100, 100), (255, 255, 255))
    
    # Get max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Calculate grid dimensions
    rows = (len(images) + columns - 1) // columns
    grid_width = columns * (max_width + padding) + padding
    grid_height = rows * (max_height + padding) + padding
    
    # Create grid
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // columns
        col = i % columns
        x = col * (max_width + padding) + padding
        y = row * (max_height + padding) + padding
        grid.paste(img, (x, y))
    
    return grid


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    try:
        import config
        
        cache = ISLImageCache(config.RAW_DATA_DIR, config.DISPLAY_SIZE)
        count = cache.load_all_images(config.SUPPORTED_SIGNS)
        
        print(f"Loaded {count} images")
        print(f"Available signs: {cache.available_signs}")
        
        # Test getting an image
        test_char = 'a'
        img = cache.get_image(test_char)
        if img:
            print(f"Successfully loaded image for '{test_char}': {img.size}")
        
    except Exception as e:
        print(f"Test error: {e}")
