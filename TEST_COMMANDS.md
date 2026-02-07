# IsharaAI - Testing Commands

## 🧪 Complete Testing Guide

### 1️⃣ Setup Virtual Environment
```bash
cd /home/karthik/IsharaAI/IsharaAI

# Activate virtual environment
source /home/karthik/IsharaAI/.venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.12.x
```

### 2️⃣ Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify key packages
pip list | grep -E "opencv|mediapipe|tensorflow|pyttsx3|tkinter"
```

### 3️⃣ Check Model Files

#### Check if models exist:
```bash
# Main application models
ls -lh models/model.h5  # ISL classifier (should be 11MB)
ls -lh models/hand_landmarker.task  # MediaPipe (should be 7.5MB)
ls -lh models/vosk-model-small-en-us-0.15/  # Speech recognition

# Standalone detector models
ls -lh isl_github_model/model.h5  # ISL classifier
ls -lh isl_github_model/hand_landmarker.task  # MediaPipe
```

#### Download missing models:
```bash
# If models are missing, run:
python setup_models.py

# Or download manually:
cd isl_github_model
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# For model.h5, download from:
# https://github.com/MaitreeVaria/Indian-Sign-Language-Detection
```

### 4️⃣ Test Individual Components

#### Test Camera Access:
```bash
python scripts/test_camera.py
# Should open webcam window - press ESC to close
```

#### Test Hand Tracking:
```bash
python tests/test_hand_tracking.py
# Should detect hands and show landmarks
```

#### Test Text-to-Speech:
```bash
python -c "
from src.core.text_to_speech import TextToSpeech
tts = TextToSpeech()
tts.speak('Hello from IsharaAI')
"
# Should hear computer speak
```

#### Test Speech Recognition:
```bash
python tests/test_speech_recognition.py
# Should record and transcribe your voice
```

### 5️⃣ Test Main Application

#### Run Full GUI Application:
```bash
python app.py
```

**What to test in GUI:**
1. **Text → ISL Tab:**
   - Type "HELLO" and press Enter
   - Should show ISL sign images

2. **Speech → ISL Tab:**
   - Click "Record"
   - Speak a word
   - Should show ISL signs

3. **ISL → Speech Tab:**
   - Allow camera access
   - Show hand signs (A-Z, 1-9)
   - Press SPACE to add letters
   - Press S to speak the sentence

### 6️⃣ Test Standalone ISL Detector (Your Improved Version)

#### Run Stable Detector with Speech:
```bash
cd isl_github_model
/home/karthik/IsharaAI/.venv/bin/python run_detector_stable.py
```

**Controls:**
- Show hand sign and hold steady
- Wait for "STABLE" indicator (green text)
- Press **SPACE** to add letter
- Press **ENTER** to add space
- Press **S** to speak sentence
- Press **C** to clear
- Press **ESC** to exit

**What to check:**
- ✅ Camera opens
- ✅ Hand landmarks appear (green lines/dots)
- ✅ Top 3 predictions show on right side
- ✅ Confidence percentage displays
- ✅ Text-to-speech works when pressing S
- ✅ Both hands are detected

### 7️⃣ Test Diagnostic Tool (For Problematic Letters)

```bash
cd isl_github_model
/home/karthik/IsharaAI/.venv/bin/python diagnose_letters.py
```

**What this shows:**
- TOP 10 predictions for current hand sign
- Where letters I and T rank
- Numbered landmarks (0-20) on hand
- Helps debug which signs are confused

### 8️⃣ Quick Verification Checklist

Run these commands to verify everything:

```bash
# Check Python environment
python --version

# Check installed packages
pip list | grep -i "opencv\|mediapipe\|tensorflow\|pyttsx3"

# Check model files
ls -lh models/model.h5 models/hand_landmarker.task
ls -lh isl_github_model/model.h5 isl_github_model/hand_landmarker.task

# Test import of core modules
python -c "
import cv2
import mediapipe as mp
import tensorflow as tf
import pyttsx3
print('✓ All imports successful')
"

# Quick camera test
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
print('✓ Camera working' if ret else '✗ Camera failed')
"
```

### 9️⃣ Troubleshooting

#### If camera doesn't work:
```bash
# Check camera devices
ls -l /dev/video*

# Test with simple OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

#### If speech doesn't work:
```bash
# Check audio devices
aplay -l  # List playback devices
arecord -l  # List recording devices

# Test pyttsx3
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"
```

#### If models are missing:
```bash
# Re-run model download
python setup_models.py

# Or manually download from GitHub repo
```

#### If MediaPipe errors:
```bash
# Check MediaPipe version
pip show mediapipe  # Should be 0.10.32

# Reinstall if needed
pip uninstall mediapipe
pip install mediapipe==0.10.32
```

### 🎯 Expected Results

✅ **Working Features:**
1. Main GUI opens without errors
2. Camera feed shows in ISL → Speech tab
3. Hand landmarks appear when showing hand
4. Letters are detected with confidence scores
5. Text-to-speech speaks sentences
6. Speech-to-text transcribes voice
7. Text-to-ISL shows sign images
8. Standalone detector works with both hands

✅ **Performance:**
- Detection latency: < 100ms per frame
- Confidence: 60%+ for good signs
- Stability: 6 frames for confirmation
- Both hands supported

### 📊 Performance Test

```bash
# Test detection speed
cd isl_github_model
/home/karthik/IsharaAI/.venv/bin/python -c "
import time
import cv2
from tensorflow import keras

start = time.time()
model = keras.models.load_model('model.h5')
print(f'Model load time: {time.time()-start:.2f}s')

cap = cv2.VideoCapture(0)
frames = 0
start = time.time()
while frames < 100:
    ret, frame = cap.read()
    if ret: frames += 1
    if time.time() - start > 3: break
cap.release()

fps = frames / (time.time() - start)
print(f'Camera FPS: {fps:.1f}')
"
```

### 🚀 Quick Start Command

**To run everything in one go:**
```bash
cd /home/karthik/IsharaAI/IsharaAI && \
source /home/karthik/IsharaAI/.venv/bin/activate && \
python app.py
```

**For standalone detector:**
```bash
cd /home/karthik/IsharaAI/IsharaAI/isl_github_model && \
/home/karthik/IsharaAI/.venv/bin/python run_detector_stable.py
```

---

## ✅ All Tests Passing = Production Ready! 🎉
