"""
Module: app.py
Description: Main Tkinter application for ISL Translation System
Author: Hackathon Team
Date: 2026

Two-way ISL translation with a modern UI:
- Mode 1: Text/Speech → ISL Signs
- Mode 2: ISL Signs → Text/Speech
"""
import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import threading
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

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available")

# Try to import MediaPipe and Keras for ISL detection
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
    
    # Load the ISL classifier model
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

# ISL alphabet (1-9, A-Z)
ISL_ALPHABET = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)


class ISLTranslatorApp:
    """
    Main application class for ISL Translation System.
    
    Provides a tabbed interface for:
    1. Text → ISL: Type text and see ISL signs
    2. Speech → ISL: Speak and see ISL signs
    3. ISL → Speech: Show signs and hear text
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ISL Translation System - Professional Edition")
        self.root.geometry("1200x750")
        self.root.configure(bg=config.COLORS['bg_primary'])
        self.root.resizable(True, True)  # Allow resizing
        
        # Initialize pipelines
        self.text_to_isl = None
        self.speech_to_isl = None
        self.tts = None
        self.image_cache = None
        
        # State variables
        self.current_signs = []
        self.current_sign_index = 0
        self.is_playing = False
        self.play_thread = None
        
        # Camera state (for ISL → Speech)
        self.camera = None
        self.is_camera_running = False
        self._frame_count = 0  # Frame counter for optimization
        self._prediction_skip = 2  # Only predict every 3rd frame (reduces lag)
        
        # Hand detection (MediaPipe Tasks API)
        self.hand_landmarker = None
        
        # ISL detection state
        self._detected_word = ""
        self._current_letter = ""
        self._letter_hold_count = 0
        self._last_letter = ""
        self._prediction_buffer = []
        self._debounce_threshold = 12  # frames to hold before accepting
        
        # Build UI
        self._setup_styles()
        self._build_ui()
        
        # Initialize pipelines in background
        self._init_pipelines_async()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_styles(self):
        """Configure ttk styles for a modern professional look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TNotebook', 
                       background=config.COLORS['bg_primary'],
                       borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=config.COLORS['bg_secondary'],
                       foreground=config.COLORS['text_secondary'],
                       padding=[24, 12],
                       font=('Segoe UI', 11, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', config.COLORS['bg_tertiary'])],
                 foreground=[('selected', config.COLORS['accent'])])
        
        style.configure('TFrame', background=config.COLORS['bg_primary'])
        
        style.configure('TLabel', 
                       background=config.COLORS['bg_primary'],
                       foreground=config.COLORS['text_primary'],
                       font=('Segoe UI', 11))
        
        style.configure('TButton',
                       background=config.COLORS['accent'],
                       foreground='white',
                       padding=[20, 12],
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       focuscolor='none')
        style.map('TButton',
                 background=[('active', config.COLORS['accent_hover'])],
                 foreground=[('active', 'white')])
        
        # Accent button style
        style.configure('Accent.TButton',
                       background=config.COLORS['error'],
                       foreground='white',
                       padding=[20, 12],
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Header.TLabel',
                       font=('Segoe UI', 24, 'bold'),
                       foreground=config.COLORS['text_primary'],
                       background=config.COLORS['bg_primary'])
        
        style.configure('Subheader.TLabel',
                       font=('Segoe UI', 11),
                       foreground=config.COLORS['text_secondary'],
                       background=config.COLORS['bg_primary'])
        
        style.configure('Status.TLabel',
                       font=('Segoe UI', 10),
                       foreground=config.COLORS['text_secondary'],
                       background=config.COLORS['bg_primary'])
        
        style.configure('Info.TLabel',
                       font=('Segoe UI', 10),
                       foreground=config.COLORS['text_secondary'],
                       background=config.COLORS['bg_secondary'],
                       padding=[15, 10])
        
        style.configure('Card.TFrame',
                       background=config.COLORS['card_bg'],
                       relief='flat',
                       borderwidth=1)
    
    def _build_ui(self):
        """Build the main UI layout."""
        # Main container
        main_container = ttk.Frame(self.root, padding="15")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header Section
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        header = ttk.Label(header_frame, 
                          text="ISL Translation System",
                          style='Header.TLabel')
        header.pack()
        
        subheader = ttk.Label(header_frame,
                             text="Bidirectional Indian Sign Language Translation",
                             style='Subheader.TLabel')
        subheader.pack(pady=(5, 0))
        
        # Tab notebook
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create tabs
        self._create_text_to_isl_tab()
        self._create_speech_to_isl_tab()
        self._create_isl_to_speech_tab()
        
        # Status bar
        status_frame = ttk.Frame(main_container)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(status_frame, 
                              textvariable=self.status_var,
                              style='Status.TLabel')
        status_bar.pack(side=tk.LEFT)
        
        # Version label
        version_label = ttk.Label(status_frame,
                                 text="v1.0 | Professional Edition",
                                 style='Status.TLabel')
        version_label.pack(side=tk.RIGHT)
    
    def _create_text_to_isl_tab(self):
        """Create the Text → ISL tab."""
        tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab, text="◉ Text → ISL")
        
        # Instructions card
        info_card = ttk.Frame(tab, style='Card.TFrame')
        info_card.pack(fill=tk.X, pady=(0, 15))
        
        info = ttk.Label(info_card,
                        text="ⓘ Type text below and press Enter or click Translate to see ISL signs\n" +
                             "Supports letters A-Z and numbers 1-9",
                        style='Info.TLabel',
                        justify=tk.CENTER)
        info.pack(padx=20, pady=15)
        
        # Input section
        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(input_frame, 
                 text="Enter text to translate:",
                 font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, pady=(0, 8))
        
        input_row = ttk.Frame(input_frame)
        input_row.pack(fill=tk.X)
        
        self.text_input = ttk.Entry(input_row, width=60, font=('Segoe UI', 12))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.text_input.bind('<Return>', lambda e: self._translate_text())
        
        translate_btn = ttk.Button(input_row, text="→ Translate", 
                                  command=self._translate_text)
        translate_btn.pack(side=tk.LEFT)
        
        # Display section
        display_frame = ttk.Frame(tab)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current sign display
        self.current_sign_label = ttk.Label(display_frame, 
                                           text="Sign will appear here",
                                           font=('Segoe UI', 16, 'bold'),
                                           foreground=config.COLORS['accent'])
        self.current_sign_label.pack(pady=(0, 10))
        
        # Image canvas with modern styling
        canvas_container = ttk.Frame(display_frame, style='Card.TFrame')
        canvas_container.pack(pady=(0, 15))
        
        self.sign_canvas = tk.Canvas(canvas_container,
                                    width=280, height=280,
                                    bg=config.COLORS['bg_primary'],
                                    highlightthickness=3,
                                    highlightbackground=config.COLORS['accent'])
        self.sign_canvas.pack(padx=10, pady=10)
        
        # Sign grid (thumbnails)
        self.sign_grid_frame = ttk.Frame(display_frame)
        self.sign_grid_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Controls
        control_frame = ttk.Frame(display_frame)
        control_frame.pack(pady=10)
        
        self.prev_btn = ttk.Button(control_frame, text="◀ Previous",
                                  command=self._prev_sign)
        self.prev_btn.pack(side=tk.LEFT, padx=8)
        
        self.play_btn = ttk.Button(control_frame, text="▶ Play All",
                                  command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=8)
        
        self.next_btn = ttk.Button(control_frame, text="Next ▶",
                                  command=self._next_sign)
        self.next_btn.pack(side=tk.LEFT, padx=8)
        
        # Speak button
        speak_btn = ttk.Button(control_frame, text="♪ Speak Text",
                              command=self._speak_text)
        speak_btn.pack(side=tk.LEFT, padx=25)
    
    def _create_speech_to_isl_tab(self):
        """Create the Speech → ISL tab."""
        tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab, text="♫ Speech → ISL")
        
        # Instructions card
        info_card = ttk.Frame(tab, style='Card.TFrame')
        info_card.pack(fill=tk.X, pady=(0, 20))
        
        info = ttk.Label(info_card,
                        text="ⓘ Click the Record button and speak clearly for 5 seconds\n" +
                             "Your speech will be transcribed and converted to ISL signs\n" +
                             "Make sure your microphone is enabled",
                        style='Info.TLabel',
                        justify=tk.CENTER)
        info.pack(padx=20, pady=15)
        
        # Record button section
        record_frame = ttk.Frame(tab)
        record_frame.pack(pady=(0, 25))
        
        self.record_btn = ttk.Button(record_frame, text="● Start Recording (5 sec)",
                                    command=self._start_recording)
        self.record_btn.pack()
        
        # Recognized text display
        text_card = ttk.Frame(tab, style='Card.TFrame')
        text_card.pack(fill=tk.X, pady=(0, 25))
        
        ttk.Label(text_card,
                 text="Recognized Speech:",
                 font=('Segoe UI', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(15, 5))
        
        self.speech_text_var = tk.StringVar(value="Click 'Start Recording' to begin...")
        speech_label = ttk.Label(text_card, 
                                textvariable=self.speech_text_var,
                                font=('Segoe UI', 12),
                                foreground=config.COLORS['accent'],
                                wraplength=700)
        speech_label.pack(anchor=tk.W, padx=20, pady=(0, 15))
        
        # Sign display section
        display_label = ttk.Label(tab,
                                 text="ISL Sign Translation:",
                                 font=('Segoe UI', 13, 'bold'))
        display_label.pack(pady=(0, 15))
        
        canvas_container = ttk.Frame(tab, style='Card.TFrame')
        canvas_container.pack(pady=(0, 15))
        
        self.speech_sign_canvas = tk.Canvas(canvas_container,
                                           width=280, height=280,
                                           bg=config.COLORS['bg_primary'],
                                           highlightthickness=3,
                                           highlightbackground=config.COLORS['accent'])
        self.speech_sign_canvas.pack(padx=10, pady=10)
        
        # Sign sequence display
        self.speech_sign_grid = ttk.Frame(tab)
        self.speech_sign_grid.pack(fill=tk.X, pady=(0, 10))
    
    def _create_isl_to_speech_tab(self):
        """Create the ISL → Speech tab."""
        tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab, text="◉ ISL → Speech")
        
        # Instructions card
        info_card = ttk.Frame(tab, style='Card.TFrame')
        info_card.pack(fill=tk.X, pady=(0, 20))
        
        info = ttk.Label(info_card,
                        text="ⓘ Show hand signs to your camera (letters A-Z, numbers 1-9)\n" +
                             "Hold each sign steady for recognition | Use buttons below to manage detected text\n" +
                             "Press Space to add spaces between words | Click Speak to hear the word",
                        style='Info.TLabel',
                        justify=tk.CENTER)
        info.pack(padx=20, pady=15)
        
        # Controls Frame (Top)
        control_frame = ttk.Frame(tab)
        control_frame.pack(pady=(0, 15))
        
        self.camera_btn = ttk.Button(control_frame, text="◉ Start Camera",
                                    command=self._toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=8)
        
        clear_btn = ttk.Button(control_frame, text="✕ Clear Text",
                              command=self._clear_camera_word)
        clear_btn.pack(side=tk.LEFT, padx=8)
        
        speak_word_btn = ttk.Button(control_frame, text="♪ Speak Word",
                                   command=self._speak_camera_word)
        speak_word_btn.pack(side=tk.LEFT, padx=8)
        
        # Add Space button
        space_btn = ttk.Button(control_frame, text="[ ] Add Space",
                              command=self._add_space_to_word)
        space_btn.pack(side=tk.LEFT, padx=8)
        
        # Add Exit button
        exit_btn = ttk.Button(control_frame, text="✕ Exit App",
                             command=self._quit_app,
                             style='Accent.TButton')
        exit_btn.pack(side=tk.LEFT, padx=15)

        # Bind space key
        self.root.bind('<space>', self._add_space_to_word)

        # Status bar
        status_frame = ttk.Frame(tab)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.camera_status_var = tk.StringVar(value="Camera: OFF")
        camera_status = ttk.Label(status_frame,
                                 textvariable=self.camera_status_var,
                                 font=('Segoe UI', 11, 'bold'),
                                 foreground=config.COLORS['warning'])
        camera_status.pack(side=tk.LEFT)

        # Camera frame with professional border
        camera_container = ttk.Frame(tab, style='Card.TFrame')
        camera_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Square canvas for better visibility
        self.camera_canvas = tk.Canvas(camera_container,
                                       width=640, height=640,
                                       bg=config.COLORS['camera_bg'],
                                       highlightthickness=3,
                                       highlightbackground=config.COLORS['accent'])
        self.camera_canvas.pack(padx=10, pady=10)
        
        # Detected text display with card styling
        detected_card = ttk.Frame(tab, style='Card.TFrame')
        detected_card.pack(fill=tk.X, pady=(10, 0))
        
        detected_frame = ttk.Frame(detected_card)
        detected_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Label(detected_frame, 
                 text="Detected Text:", 
                 font=('Segoe UI', 13, 'bold')).pack(side=tk.LEFT, padx=(0, 15))
        
        self.detected_text_var = tk.StringVar(value="(No text yet)")
        detected_label = ttk.Label(detected_frame,
                                  textvariable=self.detected_text_var,
                                  font=('Segoe UI', 15, 'bold'),
                                  foreground=config.COLORS['accent'])
        detected_label.pack(side=tk.LEFT)
    
    def _init_pipelines_async(self):
        """Initialize pipelines in background thread."""
        def init():
            try:
                # Text to ISL
                self.status_var.set("Loading image cache...")
                self.text_to_isl = TextToISLPipeline()
                
                # TTS
                if PYTTSX3_AVAILABLE:
                    self.tts = TextToSpeech()
                
                # Image cache for direct use
                self.image_cache = ISLImageCache(
                    config.RAW_DATA_DIR,
                    config.DISPLAY_SIZE
                )
                self.image_cache.load_all_images(config.SUPPORTED_SIGNS)
                
                self.status_var.set(f"Ready! {self.image_cache.loaded_count} signs loaded")
                
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
            messagebox.showwarning("Not Ready", "Please wait for initialization")
            return
        
        # Translate
        self.current_signs = self.text_to_isl.translate(text)
        self.current_sign_index = 0
        
        if not self.current_signs:
            self.current_sign_label.config(text="No translatable characters")
            return
        
        # Update display
        self._update_sign_display()
        self._update_sign_grid()
    
    def _update_sign_display(self):
        """Update the main sign display."""
        if not self.current_signs:
            return
        
        sign = self.current_signs[self.current_sign_index]
        char = sign['char']
        
        # Update label
        self.current_sign_label.config(
            text=f"'{char.upper()}' ({self.current_sign_index + 1}/{len(self.current_signs)})"
        )
        
        # Update image
        if sign['image']:
            # Convert PIL image to PhotoImage
            img = sign['image'].copy()
            img = img.resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Store reference
            self._current_photo = photo
            
            # Update canvas
            self.sign_canvas.delete("all")
            self.sign_canvas.create_image(150, 150, image=photo)
    
    def _update_sign_grid(self):
        """Update the sign thumbnail grid."""
        # Clear existing
        for widget in self.sign_grid_frame.winfo_children():
            widget.destroy()
        
        if not self.current_signs:
            return
        
        # Show up to 10 thumbnails
        for i, sign in enumerate(self.current_signs[:10]):
            frame = ttk.Frame(self.sign_grid_frame)
            frame.pack(side=tk.LEFT, padx=2)
            
            # Create thumbnail
            if sign['image']:
                thumb = sign['image'].copy()
                thumb = thumb.resize((50, 50), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                
                label = tk.Label(frame, image=photo)
                label.image = photo  # Keep reference
                label.pack()
                
                # Click to select
                label.bind('<Button-1>', lambda e, idx=i: self._select_sign(idx))
            
            # Character label
            char_label = ttk.Label(frame, text=sign['char'].upper())
            char_label.pack()
    
    def _select_sign(self, index: int):
        """Select a specific sign."""
        self.current_sign_index = index
        self._update_sign_display()
    
    def _prev_sign(self):
        """Show previous sign."""
        if self.current_signs and self.current_sign_index > 0:
            self.current_sign_index -= 1
            self._update_sign_display()
    
    def _next_sign(self):
        """Show next sign."""
        if self.current_signs and self.current_sign_index < len(self.current_signs) - 1:
            self.current_sign_index += 1
            self._update_sign_display()
    
    def _toggle_play(self):
        """Toggle automatic playback."""
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            self._play_signs()
    
    def _play_signs(self):
        """Play through signs automatically."""
        def play_loop():
            import time
            while self.is_playing and self.current_sign_index < len(self.current_signs) - 1:
                time.sleep(config.SIGN_DISPLAY_TIME / 1000)
                if self.is_playing:
                    self.current_sign_index += 1
                    self.root.after(0, self._update_sign_display)
            
            self.is_playing = False
            self.root.after(0, lambda: self.play_btn.config(text="▶ Play"))
        
        self.play_thread = threading.Thread(target=play_loop, daemon=True)
        self.play_thread.start()
    
    def _speak_text(self):
        """Speak the current text."""
        text = self.text_input.get().strip()
        if text and self.tts:
            self.tts.speak_async(text)
    
    def _start_recording(self):
        """Start speech recording."""
        self.record_btn.config(state=tk.DISABLED)
        self.speech_text_var.set("🎤 Listening... Speak now!")
        
        # Clear previous signs
        self.speech_sign_canvas.delete("all")
        for widget in self.speech_sign_grid.winfo_children():
            widget.destroy()
        
        def record():
            try:
                # Initialize speech recognizer if needed
                from src.core.speech_recognition import create_recognizer
                recognizer = create_recognizer()
                
                # Force re-check availability if needed
                if not recognizer.is_available:
                     # Try to re-initialize or check if sounddevice is actually working
                     pass
                
                if not recognizer.is_available:
                    self.root.after(0, lambda: self.speech_text_var.set("❌ No speech recognizer available"))
                    return
                
                # Recognize speech
                text = recognizer.recognize_from_microphone(duration=5)
                
                if text:
                    self.root.after(0, lambda: self.speech_text_var.set(f"Recognized: \"{text}\""))
                    
                    # Translate to ISL
                    if self.text_to_isl:
                        signs = self.text_to_isl.translate(text)
                        
                        if signs:
                            # Store for display
                            self._speech_signs = signs
                            self._speech_sign_index = 0
                            
                            # Update display on main thread
                            self.root.after(0, self._update_speech_signs_display)
                            
                            # Auto-play the signs
                            self.root.after(500, self._play_speech_signs)
                        else:
                            self.root.after(0, lambda: self.speech_text_var.set(f"Recognized: \"{text}\" (no translatable characters)"))
                else:
                    self.root.after(0, lambda: self.speech_text_var.set("No speech detected. Try again."))
                    
            except Exception as e:
                logger.error(f"Recording error: {e}")
                self.root.after(0, lambda: self.speech_text_var.set(f"Error: {e}"))
            finally:
                self.root.after(0, lambda: self.record_btn.config(state=tk.NORMAL))
        
        threading.Thread(target=record, daemon=True).start()
    
    def _update_speech_signs_display(self):
        """Update the speech tab sign display."""
        if not hasattr(self, '_speech_signs') or not self._speech_signs:
            return
        
        signs = self._speech_signs
        index = self._speech_sign_index
        
        if index >= len(signs):
            return
        
        sign = signs[index]
        
        # Update main canvas
        if sign['image']:
            img = sign['image'].copy()
            img = img.resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._speech_photo = photo
            
            self.speech_sign_canvas.delete("all")
            self.speech_sign_canvas.create_image(150, 150, image=photo)
        
        # Update grid thumbnails
        for widget in self.speech_sign_grid.winfo_children():
            widget.destroy()
        
        signs = self._speech_signs
        for i, s in enumerate(signs[:15]):
            frame = ttk.Frame(self.speech_sign_grid)
            frame.pack(side=tk.LEFT, padx=2)
            
            if s['image']:
                thumb = s['image'].copy()
                thumb = thumb.resize((40, 40), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                
                label = tk.Label(frame, image=photo, 
                               highlightthickness=2 if i == index else 0,
                               highlightbackground=config.COLORS['accent'] if i == index else 'white')
                label.image = photo
                label.pack()
            
            char_label = ttk.Label(frame, text=s['char'].upper(), 
                                  font=('Segoe UI', 8))
            char_label.pack()
    
    def _play_speech_signs(self):
        """Auto-play through speech signs."""
        if not hasattr(self, '_speech_signs') or not self._speech_signs:
            return
        
        if self._speech_sign_index < len(self._speech_signs):
            self._update_speech_signs_display()
            self._speech_sign_index += 1
            self.root.after(800, self._play_speech_signs)  # 800ms per sign
    
    def _add_space_to_word(self, event=None):
        """Add a space to the detected word."""
        # Check if ISL -> Speech tab is active
        try:
            current_tab = self.notebook.index("current")
            if current_tab != 2:  # Assuming ISL -> Speech is the 3rd tab
                return
        except tk.TclError:
            return

        if self._detected_word and not self._detected_word.endswith(" "):
            self._detected_word += " "
            self.detected_text_var.set(self._detected_word.upper())
    
    def _toggle_camera(self):
        """Toggle camera for ISL → Speech."""
        # Unfocus button to prevent spacebar from triggering it again
        self.root.focus_set()
        
        if self.is_camera_running:
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        """Start the camera for gesture recognition."""
        if not OPENCV_AVAILABLE:
            messagebox.showerror("Error", "OpenCV not installed")
            return
        
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe not installed.\nInstall with: pip install mediapipe")
            return
        
        # Check for ISL model but don't block
        if not KERAS_MODEL_AVAILABLE:
            messagebox.showinfo("ISL Model Info", 
                              "TensorFlow not available (Python 3.14).\n"
                              "Using basic rule-based gesture detection instead.\n"
                              "For full ML model, please use Python 3.9-3.11.")
        
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Initialize MediaPipe HandLandmarker (Tasks API)
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
        
        # Reset detection state
        self._detected_word = ""
        self._current_letter = ""
        self._letter_hold_count = 0
        self._last_letter = ""
        self._prediction_buffer = []
        
        self.is_camera_running = True
        self.camera_btn.config(text="⏹ Stop Camera")
        self.camera_status_var.set("Camera: ACTIVE - Show hand signs!")
        
        # Start camera loop
        self._camera_loop()
    
    def _camera_loop(self):
        """Camera update loop with ISL hand gesture detection."""
        if not self.is_camera_running:
            return
        
        ret, frame = self.camera.read()
        
        if ret:
            # Increment frame counter
            self._frame_count += 1
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detected_letter = None
            confidence = 0.0
            
            # Only run detection every N frames to reduce lag
            if self._frame_count % (self._prediction_skip + 1) == 0:
                # Create MediaPipe Image and detect
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.hand_landmarker.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        # Draw landmarks on frame
                        self._draw_hand_landmarks(rgb_frame, hand_landmarks)
                        
                        if isl_model is not None:
                            # ML Model Prediction
                            landmark_list = self._calc_landmark_list(rgb_frame, hand_landmarks)
                            processed_landmarks = self._pre_process_landmarks(landmark_list)
                            
                            import pandas as pd
                            df = pd.DataFrame(processed_landmarks).transpose()
                            predictions = isl_model.predict(df, verbose=0)
                            predicted_class = np.argmax(predictions, axis=1)
                            confidence = float(np.max(predictions))
                            
                            # Higher threshold to prevent false detections
                            if len(predicted_class) > 0 and confidence > 0.65:
                                detected_letter = ISL_ALPHABET[predicted_class[0]]
                        else:
                            # Fallback Rule-Based Prediction
                            try:
                                landmarks = self._extract_landmarks(hand_landmarks)
                                detected_letter, confidence = self._predict_letter(landmarks)
                            except Exception as e:
                                logger.error(f"Heuristic prediction error: {e}")
                
                # Update detection state
                self._process_detection(detected_letter, confidence)
            
            # Draw status on frame (every frame for smooth display)
            self._draw_status(rgb_frame)
            
            # Convert to PhotoImage (optimize resize)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 640), Image.Resampling.NEAREST)  # Match square canvas size
            photo = ImageTk.PhotoImage(img)
            
            self._camera_photo = photo
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(320, 320, image=photo)  # Center in 640x640 canvas
        
        # Schedule next update - reduced to 16ms for smoother 60fps camera display
        self.root.after(16, self._camera_loop)
    
    def _calc_landmark_list(self, image, landmarks):
        """Calculate landmark positions in pixel coordinates."""
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        for landmark in landmarks:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        
        return landmark_point
    
    def _pre_process_landmarks(self, landmark_list):
        """Pre-process landmarks for model input."""
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        
        # Flatten to 1D list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalize
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value != 0:
            temp_landmark_list = [n / max_value for n in temp_landmark_list]
        
        return temp_landmark_list
    
    def _draw_hand_landmarks(self, image, landmarks):
        """Draw hand landmarks on the image."""
        img_h, img_w = image.shape[:2]
        
        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
            (0, 17)                                    # Palm base
        ]
        
        # Draw connections
        for p1_idx, p2_idx in connections:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
            x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Draw points
        for lm in landmarks:
            cx, cy = int(lm.x * img_w), int(lm.y * img_h)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(image, (cx, cy), 2, (255, 255, 255), -1)
    
    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract and normalize hand landmarks."""
        landmarks = []
        
        # Get wrist position for normalization
        wrist = hand_landmarks.landmark[0]
        
        for lm in hand_landmarks.landmark:
            # Normalize relative to wrist
            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])
        
        return np.array(landmarks)
    
    def _predict_letter(self, landmarks: np.ndarray) -> tuple:
        """
        Predict letter from hand landmarks.
        
        Simple rule-based detection based on finger positions.
        For a full solution, use the trained model.
        """
        # Get key landmark positions
        # Thumb tip: 4, Index tip: 8, Middle tip: 12, Ring tip: 16, Pinky tip: 20
        # Finger MCPs: 5, 9, 13, 17 (base joints)
        
        # Simple heuristic: count extended fingers
        thumb_tip_y = landmarks[4*3 + 1]  # y of thumb tip
        index_tip_y = landmarks[8*3 + 1]
        middle_tip_y = landmarks[12*3 + 1]
        ring_tip_y = landmarks[16*3 + 1]
        pinky_tip_y = landmarks[20*3 + 1]
        
        index_mcp_y = landmarks[5*3 + 1]
        middle_mcp_y = landmarks[9*3 + 1]
        ring_mcp_y = landmarks[13*3 + 1]
        pinky_mcp_y = landmarks[17*3 + 1]
        
        # Count extended fingers (tip above MCP in y means extended)
        extended = 0
        if index_tip_y < index_mcp_y - 0.05: extended += 1
        if middle_tip_y < middle_mcp_y - 0.05: extended += 1
        if ring_tip_y < ring_mcp_y - 0.05: extended += 1
        if pinky_tip_y < pinky_mcp_y - 0.05: extended += 1
        
        # Thumb check (x position relative to palm)
        thumb_extended = abs(landmarks[4*3]) > 0.1
        
        # Simple gesture mapping
        if extended == 0 and not thumb_extended:
            return ('a', 0.8)  # Fist = A
        elif extended == 1 and index_tip_y < index_mcp_y:
            # Only index extended
            if middle_tip_y < middle_mcp_y - 0.03:
                return ('v', 0.8)  # Peace/V
            return ('d', 0.8)  # Pointing = D
        elif extended == 2:
            return ('v', 0.85)  # V sign
        elif extended == 3:
            return ('w', 0.8)  # W
        elif extended == 4:
            return ('b', 0.8)  # Open hand = B
        elif extended == 5 or (extended == 4 and thumb_extended):
            return ('5', 0.8)  # All fingers = 5
        elif thumb_extended and extended == 0:
            return ('a', 0.75)  # Thumbs up = A
        elif extended == 1 and pinky_tip_y < pinky_mcp_y:
            return ('i', 0.8)  # Pinky up = I
        else:
            return ('c', 0.6)  # Default C shape
        
        return (None, 0.0)
    
    def _process_detection(self, letter: str, confidence: float):
        """Process detected letter with debouncing."""
        if letter and confidence > 0.4:  # Relaxed threshold
            # Add to buffer
            self._prediction_buffer.append(letter)
            if len(self._prediction_buffer) > 5:
                self._prediction_buffer.pop(0)
            
            # Get most common prediction
            if self._prediction_buffer:
                from collections import Counter
                most_common = Counter(self._prediction_buffer).most_common(1)[0]
                detected = most_common[0]
                
                # Update current letter display
                self._current_letter = detected
                # Display detected letter (removed old gesture_var)
                
                # Debounce: require consistent detection
                if detected == self._last_letter:
                    self._letter_hold_count += 1
                else:
                    self._letter_hold_count = 1
                    self._last_letter = detected
                
                # Add to word after holding
                if self._letter_hold_count >= self._debounce_threshold:
                    self._detected_word += detected
                    self.detected_text_var.set(self._detected_word.upper())
                    
                    # Speak the letter
                    if self.tts and PYTTSX3_AVAILABLE:
                        self.tts.speak_async(detected)
                    
                    # Reset
                    self._letter_hold_count = 0
                    self._prediction_buffer = []
        else:
            # Decay hold count if nothing detected
            if self._letter_hold_count > 0:
                self._letter_hold_count -= 1
    
    def _draw_status(self, frame: np.ndarray):
        """Draw detection status on frame."""
        # Draw current letter
        cv2.putText(frame, f"Letter: {self._current_letter.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw word
        cv2.putText(frame, f"Word: {self._detected_word.upper()}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Draw hold progress
        progress = min(self._letter_hold_count / self._debounce_threshold, 1.0)
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (10, 90), (210, 110), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
        cv2.putText(frame, "Hold to confirm", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _stop_camera(self):
        """Stop the camera."""
        self.is_camera_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if self.hand_landmarker:
            self.hand_landmarker.close()
            self.hand_landmarker = None
        
        self.camera_btn.config(text="📷 Start Camera")
        self.camera_status_var.set("Camera: OFF")
        self.camera_canvas.delete("all")
    
    def _clear_camera_word(self):
        """Clear the recognized word."""
        self._detected_word = ""
        self._current_letter = ""
        self._prediction_buffer = []
        self._letter_hold_count = 0
        self.detected_text_var.set("(No text yet)")
    
    def _speak_camera_word(self):
        """Speak the recognized word."""
        word = self._detected_word
        if word and self.tts:
            self.tts.speak_async(word)
    
    def _on_close(self):
        """Handle window close."""
        self.is_playing = False
        self._stop_camera()
        
        if self.tts:
            self.tts.shutdown()
        
        self.root.destroy()
    
    def _quit_app(self):
        """Exit button handler - same as close."""
        self._on_close()
    
    def run(self):
        """Start the application."""
        logger.info("Starting ISL Translation System")
        self.root.mainloop()


def main():
    """Main entry point."""
    app = ISLTranslatorApp()
    app.run()


if __name__ == "__main__":
    main()
