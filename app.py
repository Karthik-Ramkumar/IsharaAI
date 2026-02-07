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

# Try to import Piper TTS (preferred)
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

# Try to import winsound for audio playback (Windows)
WINSOUND_AVAILABLE = False
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    pass

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
        self.root.title("ISL Translation System")
        self.root.geometry("1000x700")
        self.root.configure(bg=config.COLORS['bg_primary'])
        
        # Initialize pipelines
        self.text_to_isl = None
        self.speech_to_isl = None
        self.image_cache = None
        
        # State variables
        self.current_signs = []
        self.current_sign_index = 0
        self.is_playing = False
        self.play_thread = None
        
        # Camera state (for ISL → Speech)
        self.camera = None
        self.is_camera_running = False
        
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

        # ── Persistent TTS thread (single engine, proper COM) ─────
        self._tts_queue = queue.Queue()
        self._tts_alive = True
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # ── Dedicated ISL → Speech TTS (fresh engine per utterance) ───
        self._isl_tts_queue = queue.Queue()
        self._isl_tts_alive = True
        self._isl_tts_thread = threading.Thread(target=self._isl_tts_worker, daemon=True)
        self._isl_tts_thread.start()

    def _tts_worker(self):
        """Background thread for TTS output using Piper (or pyttsx3 fallback)."""
        while self._tts_alive:
            try:
                text = self._tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # shutdown sentinel
                break

            self._speak_with_piper(text)

    def _isl_tts_worker(self):
        """Dedicated TTS worker for the ISL → Speech tab using Piper."""
        while self._isl_tts_alive:
            try:
                text = self._isl_tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # shutdown sentinel
                break

            self._speak_with_piper(text)

    def _speak_with_piper(self, text: str) -> None:
        """Synthesize and play speech using Piper TTS (or pyttsx3 fallback)."""
        if not text:
            return

        # Use Piper TTS if available
        if PIPER_AVAILABLE and piper_voice is not None:
            try:
                # Create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    with wave.open(tmp_file, "wb") as wav_file:
                        piper_voice.synthesize(text, wav_file)

                # Play the audio file
                if WINSOUND_AVAILABLE:
                    winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                else:
                    # Fallback: try to play with system command
                    import subprocess
                    subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"], 
                                   capture_output=True, timeout=10)

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                return  # Success with Piper
            except Exception as e:
                logger.error(f"Piper TTS error: {e}")
                # Fall through to pyttsx3 fallback

        # Fallback to pyttsx3 if Piper fails or unavailable
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
        """Queue text for the persistent TTS worker thread (Text → ISL tab)."""
        if text:
            self._tts_queue.put(text)
    
    def _setup_styles(self):
        """Configure ttk styles for a modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TNotebook', background=config.COLORS['bg_primary'])
        style.configure('TNotebook.Tab', 
                       background=config.COLORS['bg_secondary'],
                       foreground=config.COLORS['text_primary'],
                       padding=[20, 10])
        style.map('TNotebook.Tab',
                 background=[('selected', config.COLORS['accent'])])
        
        style.configure('TFrame', background=config.COLORS['bg_primary'])
        style.configure('TLabel', 
                       background=config.COLORS['bg_primary'],
                       foreground=config.COLORS['text_primary'])
        style.configure('TButton',
                       background=config.COLORS['accent'],
                       foreground=config.COLORS['text_primary'],
                       padding=[15, 8])
        
        style.configure('Header.TLabel',
                       font=('Segoe UI', 24, 'bold'),
                       foreground=config.COLORS['accent'])
        
        style.configure('Status.TLabel',
                       font=('Segoe UI', 10),
                       foreground=config.COLORS['text_secondary'])
    
    def _build_ui(self):
        """Build the main UI layout."""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Label(main_container, 
                          text="ISL Translation System",
                          style='Header.TLabel')
        header.pack(pady=(0, 10))
        
        # Tab notebook
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_text_to_isl_tab()
        self._create_speech_to_isl_tab()
        self._create_isl_to_speech_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(main_container, 
                              textvariable=self.status_var,
                              style='Status.TLabel')
        status_bar.pack(pady=(10, 0))
    
    def _create_text_to_isl_tab(self):
        """Create the Text → ISL tab."""
        tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab, text="Text → ISL")
        
        # Input section
        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(input_frame, text="Enter text to translate:").pack(anchor=tk.W)
        
        input_row = ttk.Frame(input_frame)
        input_row.pack(fill=tk.X, pady=(5, 0))
        
        self.text_input = ttk.Entry(input_row, width=50, font=('Segoe UI', 14))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.text_input.bind('<Return>', lambda e: self._translate_text())
        
        translate_btn = ttk.Button(input_row, text="Translate", 
                                  command=self._translate_text)
        translate_btn.pack(side=tk.LEFT)
        
        # Display section
        display_frame = ttk.Frame(tab)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current sign display
        self.current_sign_label = ttk.Label(display_frame, 
                                           text="Enter text above",
                                           font=('Segoe UI', 18))
        self.current_sign_label.pack(pady=10)
        
        # Image canvas
        self.sign_canvas = tk.Canvas(display_frame,
                                    width=300, height=300,
                                    bg='white',
                                    highlightthickness=2,
                                    highlightbackground=config.COLORS['border'])
        self.sign_canvas.pack(pady=10)
        
        # Sign grid (thumbnails)
        self.sign_grid_frame = ttk.Frame(display_frame)
        self.sign_grid_frame.pack(fill=tk.X, pady=10)
        
        # Controls
        control_frame = ttk.Frame(display_frame)
        control_frame.pack(pady=10)
        
        self.prev_btn = ttk.Button(control_frame, text="← Previous",
                                  command=self._prev_sign)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(control_frame, text="▶ Play",
                                  command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(control_frame, text="Next →",
                                  command=self._next_sign)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Speak button
        speak_btn = ttk.Button(control_frame, text="🔊 Speak",
                              command=self._speak_text)
        speak_btn.pack(side=tk.LEFT, padx=20)
    
    def _create_speech_to_isl_tab(self):
        """Create the Speech → ISL tab."""
        tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab, text="Speech → ISL")
        
        # Info
        info = ttk.Label(tab, 
                        text="Click 'Record' and speak. Your speech will be converted to ISL signs.",
                        font=('Segoe UI', 12))
        info.pack(pady=20)
        
        # Record button
        self.record_btn = ttk.Button(tab, text="🎤 Record (5 seconds)",
                                    command=self._start_recording)
        self.record_btn.pack(pady=10)
        
        # Recognized text
        self.speech_text_var = tk.StringVar(value="Recognized text will appear here")
        speech_label = ttk.Label(tab, 
                                textvariable=self.speech_text_var,
                                font=('Segoe UI', 14),
                                wraplength=600)
        speech_label.pack(pady=20)
        
        # Sign display (reuses similar layout)
        self.speech_sign_canvas = tk.Canvas(tab,
                                           width=300, height=300,
                                           bg='white',
                                           highlightthickness=2,
                                           highlightbackground=config.COLORS['border'])
        self.speech_sign_canvas.pack(pady=10)
        
        # Sign sequence display
        self.speech_sign_grid = ttk.Frame(tab)
        self.speech_sign_grid.pack(fill=tk.X, pady=10)
    
    def _create_isl_to_speech_tab(self):
        """Create the ISL → Speech tab."""
        tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab, text="ISL → Speech")
        
        # Info
        info = ttk.Label(tab,
                        text="Show hand signs to the camera. Recognized signs will be spoken.",
                        font=('Segoe UI', 12))
        info.pack(pady=10)
        
        # Controls Frame (Top)
        control_frame = ttk.Frame(tab)
        control_frame.pack(pady=(0, 10))
        
        self.camera_btn = ttk.Button(control_frame, text="📷 Start Camera",
                                    command=self._toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(control_frame, text="Clear",
                              command=self._clear_camera_word)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        speak_word_btn = ttk.Button(control_frame, text="🔊 Speak Word",
                                   command=self._speak_camera_word)
        speak_word_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Space button
        space_btn = ttk.Button(control_frame, text="[S] Space",
                              command=self._add_space_to_word)
        space_btn.pack(side=tk.LEFT, padx=5)

        # Add Backspace button
        backspace_btn = ttk.Button(control_frame, text="[B] Backspace",
                                  command=self._backspace_word)
        backspace_btn.pack(side=tk.LEFT, padx=5)

        # Bind S key for space, B key for backspace
        self.root.bind('<s>', self._add_space_to_word)
        self.root.bind('<S>', self._add_space_to_word)
        self.root.bind('<b>', self._backspace_word_event)
        self.root.bind('<B>', self._backspace_word_event)

        # Status
        self.camera_status_var = tk.StringVar(value="Camera off")
        camera_status = ttk.Label(control_frame,
                                 textvariable=self.camera_status_var,
                                 style='Status.TLabel')
        camera_status.pack(side=tk.LEFT, padx=20)

        # Camera frame
        camera_container = ttk.Frame(tab)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        # Reduced canvas size to fit screen (4:3 aspect ratio)
        self.camera_canvas = tk.Canvas(camera_container,
                                       width=480, height=360,
                                       bg='black',
                                       highlightthickness=2,
                                       highlightbackground=config.COLORS['border'])
        self.camera_canvas.pack(pady=5)
        
        # Recognition info
        info_frame = ttk.Frame(camera_container)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.gesture_var = tk.StringVar(value="Show hand signs to detect letters")
        gesture_label = ttk.Label(info_frame,
                                 textvariable=self.gesture_var,
                                 font=('Segoe UI', 16, 'bold'))
        gesture_label.pack()
        
        self.word_var = tk.StringVar(value="")
        word_label = ttk.Label(info_frame,
                              textvariable=self.word_var,
                              font=('Segoe UI', 14))
        word_label.pack()
    
    def _init_pipelines_async(self):
        """Initialize pipelines in background thread."""
        def init():
            try:
                # Text to ISL
                self.status_var.set("Loading image cache...")
                self.text_to_isl = TextToISLPipeline()
                
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
        if text:
            self._speak_direct(text)
    
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
                        try:
                            detected_signs = self.text_to_isl.translate(text)
                            
                            if detected_signs:
                                # Store for display
                                self._speech_signs = detected_signs
                                self._speech_sign_index = 0
                                
                                # Update display on main thread
                                self.root.after(0, self._update_speech_signs_display)
                                
                                # Auto-play the signs
                                self.root.after(500, self._play_speech_signs)
                            else:
                                self.root.after(0, lambda: self.speech_text_var.set(f"Recognized: \"{text}\" (no translatable characters)"))
                        except Exception as translation_err:
                            logger.error(f"Translation error: {translation_err}")
                            self.root.after(0, lambda: self.speech_text_var.set(f"Translation Error: {translation_err}"))
                else:
                    self.root.after(0, lambda: self.speech_text_var.set("No speech detected. Try again."))
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Recording error: {error_msg}")
                self.root.after(0, lambda msg=error_msg: self.speech_text_var.set(f"Error: {msg}"))
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
            if current_tab != 2:
                return
        except tk.TclError:
            return

        if self._detected_word and not self._detected_word.endswith(" "):
            self._detected_word += " "
            self.word_var.set(f"Word: {self._detected_word.upper()}")

        if event:
            return "break"

    def _backspace_word_event(self, event=None):
        """Key-event wrapper for backspace."""
        self._backspace_word()
        if event:
            return "break"

    def _backspace_word(self):
        """Remove the last character from the detected word."""
        try:
            current_tab = self.notebook.index("current")
            if current_tab != 2:
                return
        except tk.TclError:
            return

        if self._detected_word:
            self._detected_word = self._detected_word[:-1]
            if self._detected_word:
                self.word_var.set(f"Word: {self._detected_word.upper()}")
            else:
                self.word_var.set("")
    
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
        
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            # Fallback to default backend
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
        self.camera_status_var.set("Camera running - show hand signs!")
        
        # Start camera loop
        self._camera_loop()
    
    def _camera_loop(self):
        """Camera update loop with ISL hand gesture detection."""
        if not self.is_camera_running:
            return
        
        ret, frame = self.camera.read()
        
        if ret:
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image and detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.hand_landmarker.detect(mp_image)
            
            detected_letter = None
            confidence = 0.0
            
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
                        
                        if len(predicted_class) > 0 and confidence > 0.5:
                            detected_letter = ISL_ALPHABET[predicted_class[0]]
                    else:
                        # Fallback Rule-Based Prediction
                        # Need landmarks in wrist-relative format expected by heuristic
                        try:
                            landmarks = self._extract_landmarks(hand_landmarks)
                            detected_letter, confidence = self._predict_letter(landmarks)
                        except Exception as e:
                            logger.error(f"Heuristic prediction error: {e}")
            
            # Update detection state
            self._process_detection(detected_letter, confidence)
            
            # Draw status on frame
            self._draw_status(rgb_frame)
            
            # Convert to PhotoImage
            img = Image.fromarray(rgb_frame)
            img = img.resize((480, 360), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self._camera_photo = photo
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(240, 180, image=photo)
        
        # Schedule next update (33ms ~= 30fps)
        self.root.after(33, self._camera_loop)
    
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
        """Extract and normalize hand landmarks (MediaPipe Tasks API)."""
        landmarks = []
        
        # Get wrist position for normalization
        # Tasks API: hand_landmarks is already a list of NormalizedLandmark
        wrist = hand_landmarks[0]
        
        for lm in hand_landmarks:
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
                self.gesture_var.set(f"Detected: {detected.upper()}")
                
                # Debounce: require consistent detection
                if detected == self._last_letter:
                    self._letter_hold_count += 1
                else:
                    self._letter_hold_count = 1
                    self._last_letter = detected
                
                # Add to word after holding
                if self._letter_hold_count >= self._debounce_threshold:
                    self._detected_word += detected
                    self.word_var.set(f"Word: {self._detected_word.upper()}")
                    
                    # Speak the letter via dedicated ISL TTS
                    self._isl_tts_queue.put(detected.upper())
                    
                    # Reset
                    self._letter_hold_count = 0
                    self._prediction_buffer = []
        else:
            # Decay hold count if nothing detected
            if self._letter_hold_count > 0:
                self._letter_hold_count -= 1
            if self._letter_hold_count == 0:
                self.gesture_var.set("Show hand sign...")
    
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
        self.camera_status_var.set("Camera off")
        self.camera_canvas.delete("all")
    
    def _clear_camera_word(self):
        """Clear the recognized word."""
        self._detected_word = ""
        self._current_letter = ""
        self._prediction_buffer = []
        self._letter_hold_count = 0
        self.word_var.set("")
        self.gesture_var.set("Show hand sign...")
    
    def _speak_camera_word(self):
        """Speak the full accumulated word/sentence via the ISL TTS worker."""
        word = self._detected_word.strip()
        if not word:
            self.camera_status_var.set("No word to speak — detect signs first")
            return

        self.camera_status_var.set(f"Speaking: {word}")

        # Drain any pending single-letter speaks so they don't overlap
        while not self._isl_tts_queue.empty():
            try:
                self._isl_tts_queue.get_nowait()
            except queue.Empty:
                break

        # Queue the full word for the dedicated ISL TTS worker
        self._isl_tts_queue.put(word)
    
    def _on_close(self):
        """Handle window close."""
        self.is_playing = False
        self._stop_camera()

        # Shut down persistent TTS worker
        self._tts_alive = False
        self._tts_queue.put(None)  # sentinel to unblock the thread

        # Shut down ISL TTS worker
        self._isl_tts_alive = False
        self._isl_tts_queue.put(None)

        self.root.destroy()
    
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
