"""
Module: app_ctk.py
Description: Main CustomTkinter application for ISL Translation System
Author: Hackathon Team
Date: 2026

Two-way ISL translation with a modern UI:
- Mode 1: Text/Speech → ISL Signs
- Mode 2: ISL Signs → Text/Speech

UI ONLY CHANGE: Refactored from tkinter/ttk to CustomTkinter
FUNCTIONALITY UNCHANGED: All backend logic preserved exactly
"""
import os
import sys
from pathlib import Path
import logging
import tkinter as tk
import customtkinter as ctk
import threading
import queue
import wave
import tempfile
from typing import Optional, Dict, List
from PIL import Image, ImageTk
import numpy as np
import copy
import itertools
import string

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from src.pipelines.text_to_isl import TextToISLPipeline
from src.pipelines.speech_to_isl import SpeechToISLPipeline
from src.core.text_to_speech import TextToSpeech, PYTTSX3_AVAILABLE

from src.utils.image_utils import ISLImageCache

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ============== FUNCTIONALITY UNCHANGED: TTS imports ==============
PIPER_AVAILABLE = False
piper_voice = None
try:
    from piper import PiperVoice
    PIPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "piper", "en_US-lessac-medium.onnx")
    if os.path.exists(PIPER_MODEL_PATH):
        piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
        PIPER_AVAILABLE = True
        logger.info(f"Piper TTS loaded from {PIPER_MODEL_PATH}")
    else:
        logger.warning(f"Piper model not found at {PIPER_MODEL_PATH}")
except ImportError as e:
    logger.warning(f"Piper TTS not available: {e}")
except Exception as e:
    logger.warning(f"Error loading Piper TTS: {e}")

WINSOUND_AVAILABLE = False
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    pass

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available")

MEDIAPIPE_AVAILABLE = False
KERAS_MODEL_AVAILABLE = False
isl_model = None
hand_detector = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe Tasks API available")
except ImportError as e:
    logger.warning(f"MediaPipe import error: {e}")

try:
    from tensorflow import keras
    import pandas as pd
    
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.h5")
    if os.path.exists(model_path):
        isl_model = keras.models.load_model(model_path)
        KERAS_MODEL_AVAILABLE = True
        logger.info(f"Loaded ISL model from {model_path}")
    else:
        logger.warning(f"ISL model not found at {model_path}")
except ImportError as e:
    logger.warning(f"Keras/TensorFlow import error: {e}")
except Exception as e:
    logger.warning(f"Error loading ISL model: {e}")

ISL_ALPHABET = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# UI ONLY CHANGE: CustomTkinter theme configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# UI ONLY CHANGE: Enhanced color scheme
COLORS = {
    'bg_primary': '#1a1a2e',
    'bg_secondary': '#16213e',
    'bg_card': '#0f3460',
    'text_primary': '#eaeaea',
    'text_secondary': '#a0a0a0',
    'accent': '#00a896',
    'accent_hover': '#02c39a',
    'success': '#27AE60',
    'error': '#E74C3C',
    'border': '#3a3a5c',
    'canvas_bg': '#0a0a14'
}


class ISLTranslatorApp:
    """
    Main application class for ISL Translation System.
    UI ONLY CHANGE: Uses CustomTkinter widgets
    FUNCTIONALITY UNCHANGED: All methods preserved
    """
    
    def __init__(self):
        # UI ONLY CHANGE: CTk instead of Tk
        self.root = ctk.CTk()
        self.root.title("IsharaAI - Real-Time Indian Sign Language and Speech Translation System")
        self.root.geometry("1100x750")
        self.root.configure(fg_color=COLORS['bg_primary'])
        
        # FUNCTIONALITY UNCHANGED: Pipeline initialization
        self.text_to_isl = None
        self.speech_to_isl = None
        self.image_cache = None
        
        # FUNCTIONALITY UNCHANGED: State variables
        self.current_signs = []
        self.current_sign_index = 0
        self.is_playing = False
        self.play_thread = None
        
        self.camera = None
        self.is_camera_running = False
        self.hand_landmarker = None
        self.is_recording = False
        self.speech_recognizer = None
        self._recording_stop_event = None
        
        self._detected_word = ""
        self._current_letter = ""
        self._letter_hold_count = 0
        self._last_letter = ""
        self._prediction_buffer = []
        self._debounce_threshold = 12
        
        # UI ONLY CHANGE: Build modern UI
        self._build_ui()
        
        # FUNCTIONALITY UNCHANGED
        self._init_pipelines_async()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._tts_queue = queue.Queue()
        self._tts_alive = True
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        self._isl_tts_queue = queue.Queue()
        self._isl_tts_alive = True
        self._isl_tts_thread = threading.Thread(target=self._isl_tts_worker, daemon=True)
        self._isl_tts_thread.start()

    # ============== FUNCTIONALITY UNCHANGED: TTS Workers ==============
    def _tts_worker(self):
        while self._tts_alive:
            try:
                text = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            self._speak_with_piper(text)

    def _isl_tts_worker(self):
        while self._isl_tts_alive:
            try:
                text = self._isl_tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            self._speak_with_piper(text)

    def _speak_with_piper(self, text: str) -> None:
        if not text:
            return
        if PIPER_AVAILABLE and piper_voice is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    with wave.open(tmp_file, "wb") as wav_file:
                        piper_voice.synthesize(text, wav_file)
                if WINSOUND_AVAILABLE:
                    winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                else:
                    import subprocess
                    subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"], 
                                   capture_output=True, timeout=10)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return
            except Exception as e:
                logger.error(f"Piper TTS error: {e}")
        if PYTTSX3_AVAILABLE:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', config.SPEECH_RATE)
                engine.setProperty('volume', config.SPEECH_VOLUME)
                engine.say(text)
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"pyttsx3 fallback error: {e}")

    def _speak_direct(self, text: str) -> None:
        if text:
            self._tts_queue.put(text)
    
    # ============== UI ONLY CHANGE: Build Modern UI ==============
    def _build_ui(self):
        """Build the main UI layout with CustomTkinter."""
        # Main container
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ctk.CTkFrame(main_container, fg_color=COLORS['bg_card'], corner_radius=12, height=60)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        header = ctk.CTkLabel(header_frame, 
                             text="🤟 IsharaAI - Real-Time Indian Sign Language and Speech Translation System",
                             font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                             text_color=COLORS['accent'])
        header.pack(pady=15)
        
        # Tab container
        self.tabview = ctk.CTkTabview(main_container, 
                                      fg_color=COLORS['bg_secondary'],
                                      segmented_button_fg_color=COLORS['bg_card'],
                                      segmented_button_selected_color=COLORS['accent'],
                                      segmented_button_selected_hover_color=COLORS['accent_hover'],
                                      segmented_button_unselected_color=COLORS['bg_card'],
                                      segmented_button_unselected_hover_color=COLORS['border'],
                                      corner_radius=12)
        self.tabview.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create tabs
        self.tabview.add("Translator")
        self.tabview.add("Gestures")
        self.tabview.add("Text → ISL")
        self.tabview.add("Speech → ISL")
        self.tabview.add("ISL → Speech")
        
        # Build tab contents
        self._create_translator_tab()
        self._create_cheatsheet_tab()
        self._create_text_to_isl_tab()
        self._create_speech_to_isl_tab()
        self._create_isl_to_speech_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ctk.CTkLabel(main_container, 
                                  textvariable=self.status_var,
                                  font=ctk.CTkFont(size=11),
                                  text_color=COLORS['text_secondary'])
        status_bar.pack(pady=(5, 0))

    def _create_text_to_isl_tab(self):
        """Create the Text → ISL tab with CustomTkinter."""
        tab = self.tabview.tab("Text → ISL")
        
        # Card container
        card = ctk.CTkFrame(tab, fg_color=COLORS['bg_card'], corner_radius=10)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input section
        input_frame = ctk.CTkFrame(card, fg_color="transparent")
        input_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        ctk.CTkLabel(input_frame, text="Enter text to translate:", 
                     font=ctk.CTkFont(size=14),
                     text_color=COLORS['text_primary']).pack(anchor=tk.W)
        
        input_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_row.pack(fill=tk.X, pady=(8, 0))
        
        self.text_input = ctk.CTkEntry(input_row, height=45, 
                                       font=ctk.CTkFont(size=14),
                                       fg_color=COLORS['bg_secondary'],
                                       border_color=COLORS['border'],
                                       text_color=COLORS['text_primary'])
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.text_input.bind('<Return>', lambda e: self._translate_text())
        
        translate_btn = ctk.CTkButton(input_row, text="Translate", 
                                      command=self._translate_text,
                                      fg_color=COLORS['accent'],
                                      hover_color=COLORS['accent_hover'],
                                      height=45, width=120,
                                      font=ctk.CTkFont(size=14, weight="bold"))
        translate_btn.pack(side=tk.LEFT)
        
        # Display section
        display_frame = ctk.CTkFrame(card, fg_color="transparent")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Current sign label
        self.current_sign_label = ctk.CTkLabel(display_frame, 
                                               text="Enter text above",
                                               font=ctk.CTkFont(size=18, weight="bold"),
                                               text_color=COLORS['text_primary'])
        self.current_sign_label.pack(pady=10)
        
        # Image canvas (kept as tk.Canvas)
        canvas_frame = ctk.CTkFrame(display_frame, fg_color=COLORS['canvas_bg'], corner_radius=8)
        canvas_frame.pack(pady=10)
        self.sign_canvas = tk.Canvas(canvas_frame, width=300, height=300,
                                     bg=COLORS['canvas_bg'], highlightthickness=0)
        self.sign_canvas.pack(padx=4, pady=4)
        
        # Sign grid (scrollable)
        self.sign_grid_container = ctk.CTkScrollableFrame(display_frame, orientation="horizontal",
                                                          height=90, fg_color=COLORS['bg_secondary'],
                                                          corner_radius=8)
        self.sign_grid_container.pack(fill=tk.X, pady=(10, 15))
        self.sign_grid_frame = self.sign_grid_container
        
        # Controls
        control_frame = ctk.CTkFrame(display_frame, fg_color="transparent")
        control_frame.pack(pady=10)
        
        self.prev_btn = ctk.CTkButton(control_frame, text="← Previous",
                                      command=self._prev_sign, width=100,
                                      fg_color=COLORS['bg_secondary'],
                                      hover_color=COLORS['border'])
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ctk.CTkButton(control_frame, text="▶ Play",
                                      command=self._toggle_play, width=100,
                                      fg_color=COLORS['accent'],
                                      hover_color=COLORS['accent_hover'])
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ctk.CTkButton(control_frame, text="Next →",
                                      command=self._next_sign, width=100,
                                      fg_color=COLORS['bg_secondary'],
                                      hover_color=COLORS['border'])
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        speak_btn = ctk.CTkButton(control_frame, text="🔊 Speak",
                                  command=self._speak_text, width=100,
                                  fg_color=COLORS['success'],
                                  hover_color="#2ecc71")
        speak_btn.pack(side=tk.LEFT, padx=20)

    def _create_speech_to_isl_tab(self):
        """Create the Speech → ISL tab."""
        tab = self.tabview.tab("Speech → ISL")
        
        card = ctk.CTkFrame(tab, fg_color=COLORS['bg_card'], corner_radius=10)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info
        info = ctk.CTkLabel(card, text="Click 'Record' and speak. Your speech will be converted to ISL signs.",
                           font=ctk.CTkFont(size=13),
                           text_color=COLORS['text_secondary'])
        info.pack(pady=20)
        
        # Record button
        self.record_btn = ctk.CTkButton(card, text="🎤 Start Recording",
                                        command=self._toggle_recording,
                                        fg_color=COLORS['error'],
                                        hover_color="#c0392b",
                                        height=45, width=180,
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.record_btn.pack(pady=10)
        
        # Recognized text
        self.speech_text_var = tk.StringVar(value="Recognized text will appear here")
        speech_label = ctk.CTkLabel(card, textvariable=self.speech_text_var,
                                    font=ctk.CTkFont(size=14),
                                    text_color=COLORS['text_primary'],
                                    wraplength=600)
        speech_label.pack(pady=20)
        
        # Sign display canvas
        canvas_frame = ctk.CTkFrame(card, fg_color=COLORS['canvas_bg'], corner_radius=8)
        canvas_frame.pack(pady=10)
        self.speech_sign_canvas = tk.Canvas(canvas_frame, width=300, height=300,
                                            bg=COLORS['canvas_bg'], highlightthickness=0)
        self.speech_sign_canvas.pack(padx=4, pady=4)
        
        # Sign grid
        self.speech_sign_grid_container = ctk.CTkScrollableFrame(card, orientation="horizontal",
                                                                  height=90, fg_color=COLORS['bg_secondary'],
                                                                  corner_radius=8)
        self.speech_sign_grid_container.pack(fill=tk.X, padx=20, pady=(10, 20))
        self.speech_sign_grid = self.speech_sign_grid_container

    def _create_isl_to_speech_tab(self):
        """Create the ISL → Speech tab."""
        tab = self.tabview.tab("ISL → Speech")
        
        card = ctk.CTkFrame(tab, fg_color=COLORS['bg_card'], corner_radius=10)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info
        info = ctk.CTkLabel(card, text="Show hand signs to the camera. Recognized signs will be spoken.",
                           font=ctk.CTkFont(size=13),
                           text_color=COLORS['text_secondary'])
        info.pack(pady=10)
        
        # Controls Frame
        control_frame = ctk.CTkFrame(card, fg_color="transparent")
        control_frame.pack(pady=(0, 10))
        
        self.camera_btn = ctk.CTkButton(control_frame, text="📷 Start Camera",
                                        command=self._toggle_camera,
                                        fg_color=COLORS['accent'],
                                        hover_color=COLORS['accent_hover'],
                                        height=40, width=140)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ctk.CTkButton(control_frame, text="Clear",
                                  command=self._clear_camera_word,
                                  fg_color=COLORS['bg_secondary'],
                                  hover_color=COLORS['border'],
                                  height=40, width=80)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        speak_word_btn = ctk.CTkButton(control_frame, text="🔊 Speak Word",
                                       command=self._speak_camera_word,
                                       fg_color=COLORS['success'],
                                       hover_color="#2ecc71",
                                       height=40, width=120)
        speak_word_btn.pack(side=tk.LEFT, padx=5)
        
        space_btn = ctk.CTkButton(control_frame, text="[S] Space",
                                  command=self._add_space_to_word,
                                  fg_color=COLORS['bg_secondary'],
                                  hover_color=COLORS['border'],
                                  height=40, width=90)
        space_btn.pack(side=tk.LEFT, padx=5)

        backspace_btn = ctk.CTkButton(control_frame, text="[B] Backspace",
                                      command=self._backspace_word,
                                      fg_color=COLORS['bg_secondary'],
                                      hover_color=COLORS['border'],
                                      height=40, width=110)
        backspace_btn.pack(side=tk.LEFT, padx=5)

        # FUNCTIONALITY UNCHANGED: Bind keyboard shortcuts
        self.root.bind('<s>', self._add_space_to_word)
        self.root.bind('<S>', self._add_space_to_word)
        self.root.bind('<b>', self._backspace_word_event)
        self.root.bind('<B>', self._backspace_word_event)

        # Status
        self.camera_status_var = tk.StringVar(value="Camera off")
        camera_status = ctk.CTkLabel(control_frame, textvariable=self.camera_status_var,
                                     font=ctk.CTkFont(size=11),
                                     text_color=COLORS['text_secondary'])
        camera_status.pack(side=tk.LEFT, padx=20)

        # Camera canvas
        camera_container = ctk.CTkFrame(card, fg_color="transparent")
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = ctk.CTkFrame(camera_container, fg_color=COLORS['canvas_bg'], corner_radius=8)
        canvas_frame.pack(pady=5)
        self.camera_canvas = tk.Canvas(canvas_frame, width=640, height=480,
                                       bg='black', highlightthickness=0)
        self.camera_canvas.pack(padx=4, pady=4)
        
        # Recognition info
        info_frame = ctk.CTkFrame(camera_container, fg_color="transparent")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.gesture_var = tk.StringVar(value="Show hand signs to detect letters")
        gesture_label = ctk.CTkLabel(info_frame, textvariable=self.gesture_var,
                                     font=ctk.CTkFont(size=16, weight="bold"),
                                     text_color=COLORS['accent'])
        gesture_label.pack()
        
        self.word_var = tk.StringVar(value="")
        word_label = ctk.CTkLabel(info_frame, textvariable=self.word_var,
                                  font=ctk.CTkFont(size=14),
                                  text_color=COLORS['text_primary'])
        word_label.pack()
    
    def _create_cheatsheet_tab(self):
        """Create tab to display all Gestures reference image."""
        tab = self.tabview.tab("Gestures")
        
        card = ctk.CTkFrame(tab, fg_color=COLORS['bg_card'], corner_radius=10)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scroll_container = ctk.CTkScrollableFrame(card, fg_color="transparent")
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        try:
            image_path = Path(__file__).parent / "allGestures.png"
            if image_path.exists():
                img = Image.open(image_path)
                target_width = 900
                w_percent = (target_width / float(img.size[0]))
                target_height = int((float(img.size[1]) * float(w_percent)))
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = ctk.CTkLabel(scroll_container, image=photo, text="")
                label.image = photo
                label.pack(pady=10)
            else:
                ctk.CTkLabel(scroll_container, text="allGestures.png not found",
                            text_color=COLORS['error']).pack(pady=20)
        except Exception as e:
            ctk.CTkLabel(scroll_container, text=f"Error loading image: {e}",
                        text_color=COLORS['error']).pack(pady=20)

    def _create_translator_tab(self):
        """Create the unified Translator tab with ISL→Speech on left and Speech→ISL on right."""
        tab = self.tabview.tab("Translator")
        
        # Main container with two panels
        main_frame = ctk.CTkFrame(tab, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== LEFT PANEL: ISL → Speech =====
        left_panel = ctk.CTkFrame(main_frame, fg_color=COLORS['bg_card'], corner_radius=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        left_header = ctk.CTkLabel(left_panel, text="ISL → Speech", 
                                   font=ctk.CTkFont(size=14, weight="bold"),
                                   text_color=COLORS['accent'])
        left_header.pack(pady=(10, 10))
        
        # Controls
        left_control_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        left_control_frame.pack(pady=5)
        
        self.trans_camera_btn = ctk.CTkButton(left_control_frame, text="📷 Start Camera",
                                              command=self._toggle_trans_camera,
                                              fg_color=COLORS['accent'],
                                              hover_color=COLORS['accent_hover'],
                                              height=35, width=120)
        self.trans_camera_btn.pack(side=tk.LEFT, padx=3)
        
        trans_clear_btn = ctk.CTkButton(left_control_frame, text="Clear",
                                        command=self._clear_trans_word,
                                        fg_color=COLORS['bg_secondary'],
                                        hover_color=COLORS['border'],
                                        height=35, width=60)
        trans_clear_btn.pack(side=tk.LEFT, padx=3)
        
        trans_speak_btn = ctk.CTkButton(left_control_frame, text="🔊 Speak Word",
                                        command=self._speak_trans_word,
                                        fg_color=COLORS['success'],
                                        hover_color="#2ecc71",
                                        height=35, width=100)
        trans_speak_btn.pack(side=tk.LEFT, padx=3)
        
        trans_space_btn = ctk.CTkButton(left_control_frame, text="[S] Space",
                                        command=self._add_trans_space,
                                        fg_color=COLORS['bg_secondary'],
                                        hover_color=COLORS['border'],
                                        height=35, width=80)
        trans_space_btn.pack(side=tk.LEFT, padx=3)
        
        trans_backspace_btn = ctk.CTkButton(left_control_frame, text="[B] Backspace",
                                            command=self._trans_backspace,
                                            fg_color=COLORS['bg_secondary'],
                                            hover_color=COLORS['border'],
                                            height=35, width=100)
        trans_backspace_btn.pack(side=tk.LEFT, padx=3)
        
        # Camera status
        self.trans_camera_status_var = tk.StringVar(value="Camera off")
        trans_status = ctk.CTkLabel(left_panel, textvariable=self.trans_camera_status_var,
                                    font=ctk.CTkFont(size=11),
                                    text_color=COLORS['text_secondary'])
        trans_status.pack(pady=2)
        
        # Camera canvas
        canvas_frame = ctk.CTkFrame(left_panel, fg_color=COLORS['canvas_bg'], corner_radius=8)
        canvas_frame.pack(pady=5)
        self.trans_camera_canvas = tk.Canvas(canvas_frame, width=640, height=480,
                                             bg='black', highlightthickness=0)
        self.trans_camera_canvas.pack(padx=4, pady=4)
        
        # Detected gesture/word
        self.trans_gesture_var = tk.StringVar(value="Show hand sign...")
        trans_gesture_label = ctk.CTkLabel(left_panel, textvariable=self.trans_gesture_var,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           text_color=COLORS['accent'])
        trans_gesture_label.pack()
        
        self.trans_word_var = tk.StringVar(value="")
        trans_word_label = ctk.CTkLabel(left_panel, textvariable=self.trans_word_var,
                                        font=ctk.CTkFont(size=12),
                                        text_color=COLORS['text_primary'])
        trans_word_label.pack()
        
        # ===== VERTICAL DIVIDER =====
        divider = ctk.CTkFrame(main_frame, width=3, fg_color=COLORS['border'])
        divider.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # ===== RIGHT PANEL: Text/Speech → ISL =====
        right_panel = ctk.CTkFrame(main_frame, fg_color=COLORS['bg_card'], corner_radius=10)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        right_header = ctk.CTkLabel(right_panel, text="Text/Speech → ISL", 
                                    font=ctk.CTkFont(size=14, weight="bold"),
                                    text_color=COLORS['accent'])
        right_header.pack(pady=(10, 5))
        
        # Text Input Section
        input_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        input_frame.pack(fill=tk.X, padx=15, pady=5)
        
        ctk.CTkLabel(input_frame, text="Enter text:", 
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS['text_primary']).pack(anchor=tk.W)
        
        input_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_row.pack(fill=tk.X, pady=(2, 0))
        
        self.trans_text_input = ctk.CTkEntry(input_row, height=38, 
                                             font=ctk.CTkFont(size=12),
                                             fg_color=COLORS['bg_secondary'],
                                             border_color=COLORS['border'],
                                             text_color=COLORS['text_primary'])
        self.trans_text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.trans_text_input.bind('<Return>', lambda e: self._translate_trans_text())
        
        trans_translate_btn = ctk.CTkButton(input_row, text="Translate",
                                            command=self._translate_trans_text,
                                            fg_color=COLORS['accent'],
                                            hover_color=COLORS['accent_hover'],
                                            height=38, width=90)
        trans_translate_btn.pack(side=tk.LEFT)
        
        # OR Record Section
        or_label = ctk.CTkLabel(right_panel, text="─── OR ───", 
                                font=ctk.CTkFont(size=9),
                                text_color=COLORS['text_secondary'])
        or_label.pack(pady=3)
        
        self.trans_record_btn = ctk.CTkButton(right_panel, text="🎤 Start Recording",
                                              command=self._toggle_trans_recording,
                                              fg_color=COLORS['error'],
                                              hover_color="#c0392b",
                                              height=38, width=150)
        self.trans_record_btn.pack(pady=3)
        
        # Recognized/Input text display
        self.trans_speech_text_var = tk.StringVar(value="Type text or record speech")
        trans_speech_label = ctk.CTkLabel(right_panel, textvariable=self.trans_speech_text_var,
                                          font=ctk.CTkFont(size=11),
                                          text_color=COLORS['text_primary'],
                                          wraplength=300)
        trans_speech_label.pack(pady=5)
        
        # Sign display canvas
        canvas_frame2 = ctk.CTkFrame(right_panel, fg_color=COLORS['canvas_bg'], corner_radius=8)
        canvas_frame2.pack(pady=3)
        self.trans_sign_canvas = tk.Canvas(canvas_frame2, width=200, height=200,
                                           bg=COLORS['canvas_bg'], highlightthickness=0)
        self.trans_sign_canvas.pack(padx=4, pady=4)
        
        # Sign grid (scrollable)
        self.trans_sign_grid_container = ctk.CTkScrollableFrame(right_panel, orientation="horizontal",
                                                                 height=70, fg_color=COLORS['bg_secondary'],
                                                                 corner_radius=8)
        self.trans_sign_grid_container.pack(fill=tk.X, padx=15, pady=3)
        self.trans_sign_grid = self.trans_sign_grid_container
        
        # Controls: Previous/Play/Next
        trans_control_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        trans_control_frame.pack(pady=5)
        
        self.trans_prev_btn = ctk.CTkButton(trans_control_frame, text="← Prev",
                                            command=self._trans_prev_sign,
                                            fg_color=COLORS['bg_secondary'],
                                            hover_color=COLORS['border'],
                                            height=32, width=70)
        self.trans_prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.trans_play_btn = ctk.CTkButton(trans_control_frame, text="▶ Play",
                                            command=self._toggle_trans_play,
                                            fg_color=COLORS['accent'],
                                            hover_color=COLORS['accent_hover'],
                                            height=32, width=70)
        self.trans_play_btn.pack(side=tk.LEFT, padx=2)
        
        self.trans_next_btn = ctk.CTkButton(trans_control_frame, text="Next →",
                                            command=self._trans_next_sign,
                                            fg_color=COLORS['bg_secondary'],
                                            hover_color=COLORS['border'],
                                            height=32, width=70)
        self.trans_next_btn.pack(side=tk.LEFT, padx=2)
        
        trans_speak_isl_btn = ctk.CTkButton(trans_control_frame, text="🔊 Speak",
                                            command=self._speak_trans_text,
                                            fg_color=COLORS['success'],
                                            hover_color="#2ecc71",
                                            height=32, width=70)
        trans_speak_isl_btn.pack(side=tk.LEFT, padx=10)
        
        # Initialize translator-specific state
        self.trans_is_camera_running = False
        self.trans_is_recording = False
        self._trans_detected_word = ""
        self._trans_current_letter = ""
        self._trans_last_letter = ""
        self._trans_letter_hold_count = 0
        self._trans_debounce_threshold = 15
        self._trans_prediction_buffer = []
        self._trans_speech_signs = []
        self._trans_speech_sign_index = 0
        self._trans_recording_stop_event = None
        self._trans_is_playing = False

    # ============== FUNCTIONALITY UNCHANGED: All methods below ==============
    
    def _init_pipelines_async(self):
        """Initialize pipelines in background thread."""
        def init():
            try:
                self.status_var.set("Loading image cache...")
                self.text_to_isl = TextToISLPipeline()
                self.image_cache = ISLImageCache(config.RAW_DATA_DIR, config.DISPLAY_SIZE)
                self.image_cache.load_all_images(config.SUPPORTED_SIGNS)
                self.status_var.set(f"Ready! {self.image_cache.loaded_count} signs loaded")
                from src.core.speech_recognition import create_recognizer
                self.speech_recognizer = create_recognizer()
                if self.speech_recognizer.is_available:
                    logger.info("Speech recognizer ready")
                else:
                    logger.warning("Speech recognizer not available")
            except Exception as e:
                logger.error(f"Init error: {e}")
                self.status_var.set(f"Error: {e}")
        threading.Thread(target=init, daemon=True).start()
    
    def _translate_text(self):
        """Translate text input to ISL signs."""
        text = self.text_input.get().strip()
        if not text:
            return
        if not self.text_to_isl:
            from tkinter import messagebox
            messagebox.showwarning("Not Ready", "Please wait for initialization")
            return
        self.current_signs = self.text_to_isl.translate(text)
        self.current_sign_index = 0
        if not self.current_signs:
            self.current_sign_label.configure(text="No translatable characters")
            return
        self._update_sign_display()
        self._update_sign_grid()
    
    def _update_sign_display(self):
        """Update the main sign display."""
        if not self.current_signs:
            return
        sign = self.current_signs[self.current_sign_index]
        char = sign['char']
        self.current_sign_label.configure(
            text=f"'{char.upper()}' ({self.current_sign_index + 1}/{len(self.current_signs)})"
        )
        if sign['image']:
            img = sign['image'].copy()
            img = img.resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._current_photo = photo
            self.sign_canvas.delete("all")
            self.sign_canvas.create_image(150, 150, image=photo)
    
    def _update_sign_grid(self):
        """Update the sign thumbnail grid."""
        for widget in self.sign_grid_frame.winfo_children():
            widget.destroy()
        if not self.current_signs:
            return
        for i, sign in enumerate(self.current_signs):
            frame = ctk.CTkFrame(self.sign_grid_frame, fg_color="transparent")
            frame.pack(side=tk.LEFT, padx=2)
            if sign['image']:
                thumb = sign['image'].copy()
                thumb = thumb.resize((50, 50), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                label = tk.Label(frame, image=photo, bg=COLORS['bg_secondary'])
                label.image = photo
                label.pack()
                label.bind('<Button-1>', lambda e, idx=i: self._select_sign(idx))
            char_label = ctk.CTkLabel(frame, text=sign['char'].upper(), 
                                      font=ctk.CTkFont(size=10),
                                      text_color=COLORS['text_primary'])
            char_label.pack()
    
    def _select_sign(self, index: int):
        self.current_sign_index = index
        self._update_sign_display()
    
    def _prev_sign(self):
        if self.current_signs and self.current_sign_index > 0:
            self.current_sign_index -= 1
            self._update_sign_display()
    
    def _next_sign(self):
        if self.current_signs and self.current_sign_index < len(self.current_signs) - 1:
            self.current_sign_index += 1
            self._update_sign_display()
    
    def _toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.configure(text="▶ Play")
        else:
            self.is_playing = True
            self.play_btn.configure(text="⏸ Pause")
            self._play_signs()
    
    def _play_signs(self):
        def play_loop():
            import time
            while self.is_playing and self.current_sign_index < len(self.current_signs) - 1:
                time.sleep(config.SIGN_DISPLAY_TIME / 1000)
                if self.is_playing:
                    self.current_sign_index += 1
                    self.root.after(0, self._update_sign_display)
            self.is_playing = False
            self.root.after(0, lambda: self.play_btn.configure(text="▶ Play"))
        self.play_thread = threading.Thread(target=play_loop, daemon=True)
        self.play_thread.start()
    
    def _speak_text(self):
        text = self.text_input.get().strip()
        if text:
            self._speak_direct(text)

    def _toggle_recording(self):
        """Toggle speech recording on/off."""
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.configure(text="⬛ Stop Recording")
            self.speech_text_var.set("🎤 Listening... Speak now!")
            self.speech_sign_canvas.delete("all")
            for widget in self.speech_sign_grid.winfo_children():
                widget.destroy()
            self._speech_signs = []
            self._recording_stop_event = threading.Event()
            threading.Thread(target=self._recording_thread, daemon=True).start()
        else:
            self.is_recording = False
            self.record_btn.configure(text="🎤 Start Recording")
            if self._recording_stop_event:
                self._recording_stop_event.set()
    
    def _recording_thread(self):
        if not self.speech_recognizer or not self.speech_recognizer.is_available:
            self.root.after(0, lambda: self.speech_text_var.set("❌ No recognizer available"))
            self.root.after(0, self._toggle_recording)
            return
        def callback(text, is_final):
            if text:
                self.root.after(0, lambda: self.speech_text_var.set(f'Recognized: "{text}"'))
                if is_final:
                    self._process_speech_text(text)
        try:
            if hasattr(self.speech_recognizer, 'listen_continuously'):
                self.speech_recognizer.listen_continuously(callback, self._recording_stop_event)
            else:
                text = self.speech_recognizer.recognize_from_microphone(duration=5)
                if text:
                    callback(text, True)
                self.root.after(0, self._toggle_recording)
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.root.after(0, lambda: self.speech_text_var.set(f"Error: {e}"))
            self.root.after(0, self._toggle_recording)

    def _process_speech_text(self, text):
        if not text or not self.text_to_isl:
            return
        try:
            detected_signs = self.text_to_isl.translate(text)
            if detected_signs:
                self._speech_signs = detected_signs
                self._speech_sign_index = 0
                self.root.after(0, self._update_speech_signs_display)
                self.root.after(500, self._play_speech_signs)
            else:
                self.root.after(0, lambda: self.speech_text_var.set(f'Recognized: "{text}" (no translatable characters)'))
        except Exception as translation_err:
            logger.error(f"Translation error: {translation_err}")
    
    def _update_speech_signs_display(self):
        if not hasattr(self, '_speech_signs') or not self._speech_signs:
            return
        signs = self._speech_signs
        index = self._speech_sign_index
        if index >= len(signs):
            return
        sign = signs[index]
        if sign['image']:
            img = sign['image'].copy()
            img = img.resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._speech_photo = photo
            self.speech_sign_canvas.delete("all")
            self.speech_sign_canvas.create_image(150, 150, image=photo)
        for widget in self.speech_sign_grid.winfo_children():
            widget.destroy()
        for i, s in enumerate(signs):
            frame = ctk.CTkFrame(self.speech_sign_grid, fg_color="transparent")
            frame.pack(side=tk.LEFT, padx=2)
            if s['image']:
                thumb = s['image'].copy()
                thumb = thumb.resize((40, 40), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                hl = 2 if i == index else 0
                bg = COLORS['accent'] if i == index else COLORS['bg_secondary']
                label = tk.Label(frame, image=photo, highlightthickness=hl, highlightbackground=bg, bg=COLORS['bg_secondary'])
                label.image = photo
                label.pack()
            char_label = ctk.CTkLabel(frame, text=s['char'].upper(), font=ctk.CTkFont(size=8),
                                      text_color=COLORS['text_primary'])
            char_label.pack()
    
    def _play_speech_signs(self):
        if not hasattr(self, '_speech_signs') or not self._speech_signs:
            return
        if self._speech_sign_index < len(self._speech_signs):
            self._update_speech_signs_display()
            self._speech_sign_index += 1
            self.root.after(800, self._play_speech_signs)
    
    def _add_space_to_word(self, event=None):
        try:
            current_tab = self.tabview.get()
            if current_tab == "ISL → Speech":
                if self._detected_word and not self._detected_word.endswith(" "):
                    self._detected_word += " "
                    self.word_var.set(f"Word: {self._detected_word.upper()}")
            elif current_tab == "Translator":
                self._add_trans_space()
            else:
                return
        except Exception:
            return
        if event:
            return "break"

    def _backspace_word_event(self, event=None):
        self._backspace_word()
        if event:
            return "break"

    def _backspace_word(self):
        try:
            current_tab = self.tabview.get()
            if current_tab == "ISL → Speech":
                if self._detected_word:
                    self._detected_word = self._detected_word[:-1]
                    if self._detected_word:
                        self.word_var.set(f"Word: {self._detected_word.upper()}")
                    else:
                        self.word_var.set("")
            elif current_tab == "Translator":
                self._trans_backspace()
        except Exception:
            return
    
    def _toggle_camera(self):
        """Toggle camera for ISL → Speech."""
        self.root.focus_set()
        if self.is_camera_running:
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        """Start the camera for gesture recognition."""
        from tkinter import messagebox
        if not OPENCV_AVAILABLE:
            messagebox.showerror("Error", "OpenCV not installed")
            return
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe not installed.\nInstall with: pip install mediapipe")
            return
        if not KERAS_MODEL_AVAILABLE:
            messagebox.showinfo("ISL Model Info", 
                              "TensorFlow not available.\nUsing basic rule-based gesture detection instead.\nFor full ML model, please use Python 3.9-3.11.")
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        try:
            hand_model_path = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
            base_options = python.BaseOptions(model_asset_path=hand_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("HandLandmarker initialized")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize hand detector: {e}")
            self.camera.release()
            return
        self._detected_word = ""
        self._current_letter = ""
        self._letter_hold_count = 0
        self._last_letter = ""
        self._prediction_buffer = []
        self.is_camera_running = True
        self.camera_btn.configure(text="⏹ Stop Camera")
        self.camera_status_var.set("Camera running - show hand signs!")
        self._camera_loop()
    
    def _camera_loop(self):
        """Camera update loop with ISL hand gesture detection."""
        if not self.is_camera_running:
            return
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.hand_landmarker.detect(mp_image)
            detected_letter = None
            confidence = 0.0
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    self._draw_hand_landmarks(rgb_frame, hand_landmarks)
                    if isl_model is not None:
                        landmark_list = self._calc_landmark_list(rgb_frame, hand_landmarks)
                        processed_landmarks = self._pre_process_landmarks(landmark_list)
                        import pandas as pd
                        df = pd.DataFrame(processed_landmarks).transpose()
                        predictions = isl_model.predict(df, verbose=0)
                        predicted_class = np.argmax(predictions, axis=1)
                        confidence = float(np.max(predictions))
                        if len(predicted_class) > 0 and confidence > 0.5:
                            detected_letter = ISL_ALPHABET[predicted_class[0]]
                    else:
                        try:
                            landmarks = self._extract_landmarks(hand_landmarks)
                            detected_letter, confidence = self._predict_letter(landmarks)
                        except Exception as e:
                            logger.error(f"Heuristic prediction error: {e}")
            self._process_detection(detected_letter, confidence)
            self._draw_status(rgb_frame)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._camera_photo = photo
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(320, 240, image=photo)
        self.root.after(33, self._camera_loop)
    
    def _calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for landmark in landmarks:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point
    
    def _pre_process_landmarks(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value != 0:
            temp_landmark_list = [n / max_value for n in temp_landmark_list]
        return temp_landmark_list
    
    def _draw_hand_landmarks(self, image, landmarks):
        img_h, img_w = image.shape[:2]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for p1_idx, p2_idx in connections:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
            x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        for lm in landmarks:
            cx, cy = int(lm.x * img_w), int(lm.y * img_h)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(image, (cx, cy), 2, (255, 255, 255), -1)
    
    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        landmarks = []
        wrist = hand_landmarks[0]
        for lm in hand_landmarks:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(landmarks)
    
    def _predict_letter(self, landmarks: np.ndarray) -> tuple:
        thumb_tip_y = landmarks[4*3 + 1]
        index_tip_y = landmarks[8*3 + 1]
        middle_tip_y = landmarks[12*3 + 1]
        ring_tip_y = landmarks[16*3 + 1]
        pinky_tip_y = landmarks[20*3 + 1]
        index_mcp_y = landmarks[5*3 + 1]
        middle_mcp_y = landmarks[9*3 + 1]
        ring_mcp_y = landmarks[13*3 + 1]
        pinky_mcp_y = landmarks[17*3 + 1]
        extended = 0
        if index_tip_y < index_mcp_y - 0.05: extended += 1
        if middle_tip_y < middle_mcp_y - 0.05: extended += 1
        if ring_tip_y < ring_mcp_y - 0.05: extended += 1
        if pinky_tip_y < pinky_mcp_y - 0.05: extended += 1
        thumb_extended = abs(landmarks[4*3]) > 0.1
        if extended == 0 and not thumb_extended:
            return ('a', 0.8)
        elif extended == 1 and index_tip_y < index_mcp_y:
            if middle_tip_y < middle_mcp_y - 0.03:
                return ('v', 0.8)
            return ('d', 0.8)
        elif extended == 2:
            return ('v', 0.85)
        elif extended == 3:
            return ('w', 0.8)
        elif extended == 4:
            return ('b', 0.8)
        elif extended == 5 or (extended == 4 and thumb_extended):
            return ('5', 0.8)
        elif thumb_extended and extended == 0:
            return ('a', 0.75)
        elif extended == 1 and pinky_tip_y < pinky_mcp_y:
            return ('i', 0.8)
        else:
            return ('c', 0.6)
        return (None, 0.0)
    
    def _process_detection(self, letter: str, confidence: float):
        if letter and confidence > 0.4:
            self._prediction_buffer.append(letter)
            if len(self._prediction_buffer) > 5:
                self._prediction_buffer.pop(0)
            if self._prediction_buffer:
                from collections import Counter
                most_common = Counter(self._prediction_buffer).most_common(1)[0]
                detected = most_common[0]
                self._current_letter = detected
                self.gesture_var.set(f"Detected: {detected.upper()}")
                if detected == self._last_letter:
                    self._letter_hold_count += 1
                else:
                    self._letter_hold_count = 1
                    self._last_letter = detected
                if self._letter_hold_count >= self._debounce_threshold:
                    self._detected_word += detected
                    self.word_var.set(f"Word: {self._detected_word.upper()}")
                    self._isl_tts_queue.put(detected.upper())
                    self._letter_hold_count = 0
                    self._prediction_buffer = []
        else:
            if self._letter_hold_count > 0:
                self._letter_hold_count -= 1
            if self._letter_hold_count == 0:
                self.gesture_var.set("Show hand sign...")
    
    def _draw_status(self, frame: np.ndarray):
        cv2.putText(frame, f"Letter: {self._current_letter.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Word: {self._detected_word.upper()}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        progress = min(self._letter_hold_count / self._debounce_threshold, 1.0)
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (10, 90), (210, 110), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
        cv2.putText(frame, "Hold to confirm", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _stop_camera(self):
        self.is_camera_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.hand_landmarker:
            self.hand_landmarker.close()
            self.hand_landmarker = None
        self.camera_btn.configure(text="📷 Start Camera")
        self.camera_status_var.set("Camera off")
        self.camera_canvas.delete("all")
    
    def _clear_camera_word(self):
        self._detected_word = ""
        self._current_letter = ""
        self._prediction_buffer = []
        self._letter_hold_count = 0
        self.word_var.set("")
        self.gesture_var.set("Show hand sign...")
    
    def _speak_camera_word(self):
        word = self._detected_word.strip()
        if not word:
            self.camera_status_var.set("No word to speak — detect signs first")
            return
        self.camera_status_var.set(f"Speaking: {word}")
        while not self._isl_tts_queue.empty():
            try:
                self._isl_tts_queue.get_nowait()
            except queue.Empty:
                break
        self._isl_tts_queue.put(word)
    
    # ========== TRANSLATOR TAB HANDLERS (FUNCTIONALITY UNCHANGED) ==========
    
    def _toggle_trans_camera(self):
        if not self.trans_is_camera_running:
            self.trans_camera_status_var.set("Starting camera...")
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.camera.isOpened():
                self.trans_camera_status_var.set("Camera failed to open")
                return
            if self.hand_landmarker is None:
                try:
                    model_path = Path(__file__).parent / "models" / "hand_landmarker.task"
                    if not model_path.exists():
                        self.trans_camera_status_var.set("Hand model not found")
                        return
                    options = vision.HandLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=1,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
                    logger.info("HandLandmarker initialized for Translator tab")
                except Exception as e:
                    logger.error(f"Failed to init hand landmarker: {e}")
                    self.trans_camera_status_var.set(f"Error: {e}")
                    return
            self._trans_detected_word = ""
            self._trans_last_letter = ""
            self._trans_letter_count = 0
            self._trans_prediction_buffer = []
            self.trans_is_camera_running = True
            self.trans_camera_btn.configure(text="⬛ Stop Camera")
            self.trans_camera_status_var.set("Camera running")
            self._trans_camera_loop()
        else:
            self.trans_is_camera_running = False
            self.trans_camera_btn.configure(text="📷 Start Camera")
            self.trans_camera_status_var.set("Camera off")
            self.trans_camera_canvas.delete("all")
            self._trans_current_letter = ""
            self.trans_gesture_var.set("Show hand sign...")
    
    def _trans_camera_loop(self):
        if not self.trans_is_camera_running:
            return
        if self.camera is None or not self.camera.isOpened():
            return
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.hand_landmarker.detect(mp_image)
            detected_letter = None
            confidence = 0.0
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    self._draw_hand_landmarks(rgb_frame, hand_landmarks)
                    if isl_model is not None:
                        landmark_list = self._calc_landmark_list(rgb_frame, hand_landmarks)
                        processed_landmarks = self._pre_process_landmarks(landmark_list)
                        import pandas as pd
                        df = pd.DataFrame(processed_landmarks).transpose()
                        predictions = isl_model.predict(df, verbose=0)
                        predicted_class = np.argmax(predictions, axis=1)
                        confidence = float(np.max(predictions))
                        if len(predicted_class) > 0 and confidence > 0.5:
                            detected_letter = ISL_ALPHABET[predicted_class[0]]
                    else:
                        try:
                            landmarks = self._extract_landmarks(hand_landmarks)
                            detected_letter, confidence = self._predict_letter(landmarks)
                        except Exception as e:
                            logger.error(f"Heuristic error: {e}")
            self._process_trans_detection(detected_letter, confidence)
            self._draw_trans_status(rgb_frame)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._trans_camera_photo = photo
            self.trans_camera_canvas.delete("all")
            self.trans_camera_canvas.create_image(320, 240, image=photo)
        if self.trans_is_camera_running:
            self.root.after(33, self._trans_camera_loop)
    
    def _process_trans_detection(self, letter, confidence):
        if letter and confidence > 0.4:
            self._trans_prediction_buffer.append(letter)
            if len(self._trans_prediction_buffer) > 5:
                self._trans_prediction_buffer.pop(0)
            if self._trans_prediction_buffer:
                from collections import Counter
                most_common = Counter(self._trans_prediction_buffer).most_common(1)[0]
                detected = most_common[0]
                self._trans_current_letter = detected
                self.trans_gesture_var.set(f"Detected: {detected.upper()}")
                if detected == self._trans_last_letter:
                    self._trans_letter_hold_count += 1
                else:
                    self._trans_letter_hold_count = 1
                    self._trans_last_letter = detected
                if self._trans_letter_hold_count >= self._trans_debounce_threshold:
                    self._trans_detected_word += detected
                    self.trans_word_var.set(f"Word: {self._trans_detected_word.upper()}")
                    self._isl_tts_queue.put(detected.upper())
                    self._trans_letter_hold_count = 0
                    self._trans_prediction_buffer = []
        else:
            if self._trans_letter_hold_count > 0:
                self._trans_letter_hold_count -= 1
            if self._trans_letter_hold_count == 0:
                self.trans_gesture_var.set("Show hand sign...")
    
    def _draw_trans_status(self, frame):
        cv2.putText(frame, f"Letter: {self._trans_current_letter.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Word: {self._trans_detected_word.upper()}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        progress = min(self._trans_letter_hold_count / self._trans_debounce_threshold, 1.0)
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (10, 90), (210, 110), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
        cv2.putText(frame, "Hold to confirm", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _clear_trans_word(self):
        self._trans_detected_word = ""
        self.trans_word_var.set("")
        self.trans_gesture_var.set("Show hand sign...")
    
    def _speak_trans_word(self):
        if self._trans_detected_word:
            self._speak_direct(self._trans_detected_word)
    
    def _add_trans_space(self):
        if self._trans_detected_word and not self._trans_detected_word.endswith(" "):
            self._trans_detected_word += " "
            self.trans_word_var.set(f"Word: {self._trans_detected_word.upper()}")
    
    def _trans_backspace(self):
        if self._trans_detected_word:
            self._trans_detected_word = self._trans_detected_word[:-1]
            if self._trans_detected_word:
                self.trans_word_var.set(f"Word: {self._trans_detected_word.upper()}")
            else:
                self.trans_word_var.set("")
    
    def _toggle_trans_recording(self):
        if not self.trans_is_recording:
            self.trans_is_recording = True
            self.trans_record_btn.configure(text="⬛ Stop Recording")
            self.trans_speech_text_var.set("🎤 Listening...")
            self.trans_sign_canvas.delete("all")
            for widget in self.trans_sign_grid.winfo_children():
                widget.destroy()
            self._trans_speech_signs = []
            self._trans_recording_stop_event = threading.Event()
            threading.Thread(target=self._trans_recording_thread, daemon=True).start()
        else:
            self.trans_is_recording = False
            self.trans_record_btn.configure(text="🎤 Start Recording")
            if self._trans_recording_stop_event:
                self._trans_recording_stop_event.set()
    
    def _trans_recording_thread(self):
        if not self.speech_recognizer or not self.speech_recognizer.is_available:
            self.root.after(0, lambda: self.trans_speech_text_var.set("❌ No recognizer"))
            self.root.after(0, self._toggle_trans_recording)
            return
        def callback(text, is_final):
            if text:
                self.root.after(0, lambda: self.trans_speech_text_var.set(f'Recognized: "{text}"'))
                if is_final:
                    self._process_trans_speech(text)
        try:
            if hasattr(self.speech_recognizer, 'listen_continuously'):
                self.speech_recognizer.listen_continuously(callback, self._trans_recording_stop_event)
            else:
                text = self.speech_recognizer.recognize_from_microphone(duration=5)
                if text:
                    callback(text, True)
                self.root.after(0, self._toggle_trans_recording)
        except Exception as e:
            logger.error(f"Trans recording error: {e}")
            self.root.after(0, lambda: self.trans_speech_text_var.set(f"Error: {e}"))
            self.root.after(0, self._toggle_trans_recording)
    
    def _process_trans_speech(self, text):
        if not text or not self.text_to_isl:
            return
        try:
            signs = self.text_to_isl.translate(text)
            if signs:
                self._trans_speech_signs = signs
                self._trans_speech_sign_index = 0
                self.root.after(0, self._update_trans_signs_display)
                self.root.after(500, self._play_trans_signs)
        except Exception as e:
            logger.error(f"Trans translation error: {e}")
    
    def _update_trans_signs_display(self):
        if not self._trans_speech_signs:
            return
        index = self._trans_speech_sign_index
        if index >= len(self._trans_speech_signs):
            return
        sign = self._trans_speech_signs[index]
        if sign['image']:
            img = sign['image'].copy()
            img = img.resize((230, 230), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._trans_sign_photo = photo
            self.trans_sign_canvas.delete("all")
            self.trans_sign_canvas.create_image(125, 125, image=photo)
        for widget in self.trans_sign_grid.winfo_children():
            widget.destroy()
        for i, s in enumerate(self._trans_speech_signs):
            frame = ctk.CTkFrame(self.trans_sign_grid, fg_color="transparent")
            frame.pack(side=tk.LEFT, padx=2)
            if s['image']:
                thumb = s['image'].copy()
                thumb = thumb.resize((35, 35), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                hl = 2 if i == index else 0
                bg = COLORS['accent'] if i == index else COLORS['bg_secondary']
                label = tk.Label(frame, image=photo, highlightthickness=hl, highlightbackground=bg, bg=COLORS['bg_secondary'])
                label.image = photo
                label.pack()
            char_label = ctk.CTkLabel(frame, text=s['char'].upper(), font=ctk.CTkFont(size=7),
                                      text_color=COLORS['text_primary'])
            char_label.pack()
    
    def _play_trans_signs(self):
        if not self._trans_speech_signs or not self._trans_is_playing:
            return
        if self._trans_speech_sign_index < len(self._trans_speech_signs):
            self._update_trans_signs_display()
            self._trans_speech_sign_index += 1
            self.root.after(int(config.SIGN_DISPLAY_TIME), self._play_trans_signs)
        else:
            self._trans_is_playing = False
            self.trans_play_btn.configure(text="▶ Play")
    
    def _translate_trans_text(self):
        text = self.trans_text_input.get().strip()
        if not text:
            return
        if not self.text_to_isl:
            self.trans_speech_text_var.set("Translation engine loading...")
            return
        try:
            self.trans_speech_text_var.set(f'Translating: "{text}"')
            signs = self.text_to_isl.translate(text)
            if signs:
                self._trans_speech_signs = signs
                self._trans_speech_sign_index = 0
                self._update_trans_signs_display()
                self.trans_speech_text_var.set(f'Showing: "{text}"')
            else:
                self.trans_speech_text_var.set("No signs found for this text")
        except Exception as e:
            logger.error(f"Trans text translation error: {e}")
            self.trans_speech_text_var.set(f"Error: {e}")
    
    def _trans_prev_sign(self):
        if self._trans_speech_signs and self._trans_speech_sign_index > 0:
            self._trans_speech_sign_index -= 1
            self._update_trans_signs_display()
    
    def _trans_next_sign(self):
        if self._trans_speech_signs and self._trans_speech_sign_index < len(self._trans_speech_signs) - 1:
            self._trans_speech_sign_index += 1
            self._update_trans_signs_display()
    
    def _toggle_trans_play(self):
        if not self._trans_speech_signs:
            return
        if self._trans_is_playing:
            self._trans_is_playing = False
            self.trans_play_btn.configure(text="▶ Play")
        else:
            self._trans_is_playing = True
            self.trans_play_btn.configure(text="⏸ Pause")
            if self._trans_speech_sign_index >= len(self._trans_speech_signs) - 1:
                self._trans_speech_sign_index = 0
            self._play_trans_signs()
    
    def _speak_trans_text(self):
        text = self.trans_text_input.get().strip()
        if not text:
            current_text = self.trans_speech_text_var.get()
            if current_text.startswith("Showing:") or current_text.startswith("Recognized:"):
                text = current_text.split('"')[1] if '"' in current_text else ""
        if text:
            self._speak_direct(text)

    def _on_close(self):
        """Handle window close."""
        self.is_playing = False
        self._stop_camera()
        self._tts_alive = False
        self._tts_queue.put(None)
        self._isl_tts_alive = False
        self._isl_tts_queue.put(None)
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        logger.info("Starting ISL Translation System (CustomTkinter)")
        self.root.mainloop()


def main():
    """Main entry point."""
    app = ISLTranslatorApp()
    app.run()


if __name__ == "__main__":
    main()
