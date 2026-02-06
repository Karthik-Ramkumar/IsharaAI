"""
Central configuration file for ISL Translation System
Author: Hackathon Team
Date: 2026
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw', 'isl_images')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
LANDMARKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'landmarks')
ISL_DISPLAY_DIR = os.path.join(PROCESSED_DATA_DIR, 'isl_display')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
VOSK_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

# Supported ISL Signs (Letters A-Z and Numbers 1-9)
# Maps lowercase characters to folder names in the dataset
SUPPORTED_SIGNS = {
    # Letters
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E',
    'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O',
    'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y',
    'z': 'Z',
    # Numbers
    '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
    '6': '6', '7': '7', '8': '8', '9': '9'
}

# Reverse mapping (folder name to display character)
SIGN_TO_CHAR = {v: k for k, v in SUPPORTED_SIGNS.items()}

# All supported characters (for display)
SUPPORTED_CHARS = list(SUPPORTED_SIGNS.keys())

# Model Parameters
IMAGE_SIZE = (224, 224)
DISPLAY_SIZE = (150, 150)
NUM_LANDMARKS = 21
LANDMARK_FEATURES = NUM_LANDMARKS * 3  # x, y, z coordinates = 63

# Video Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 20

# ML Settings
CONFIDENCE_THRESHOLD = 0.7
PREDICTION_SMOOTHING_WINDOW = 5
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# UI Settings
WINDOW_TITLE = "ISL Translation System"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BG_COLOR = "#2C3E50"
TEXT_COLOR = "#ECF0F1"
ACCENT_COLOR = "#3498DB"
SUCCESS_COLOR = "#27AE60"
ERROR_COLOR = "#E74C3C"

# UI Colors Dictionary (for Tkinter app)
COLORS = {
    'bg_primary': '#2C3E50',
    'bg_secondary': '#34495E',
    'text_primary': '#ECF0F1',
    'text_secondary': '#BDC3C7',
    'accent': '#3498DB',
    'success': '#27AE60',
    'error': '#E74C3C',
    'border': '#7F8C8D'
}

# Display Settings
SIGN_DISPLAY_TIME = 1000  # milliseconds per sign in slideshow

# Speech Settings
SPEECH_RATE = 150
SPEECH_VOLUME = 0.9

# Logging
import logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
