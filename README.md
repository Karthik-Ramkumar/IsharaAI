# IsharaAI - Indian Sign Language (ISL) Translation System

IsharaAI is a two-way translation system designed to bridge the communication gap between the deaf/mute community and the general public using Indian Sign Language.

## Features
- **ISL → Speech**: Real-time camera detection of hand gestures (A-Z, 1-9) with Text-to-Speech output.
- **Text → ISL**: Converts typed English text into a sequence of ISL sign images.
- **Speech → ISL**: Uses voice recognition to translate spoken English into ISL signs.
- **Rule-Based Fallback**: Intelligent fallback for newer Python versions where TensorFlow isn't yet available.

## Prerequisites
- **Python 3.9** (Recommended for full ML support)
- Webcam (for ISL → Speech)
- Microphone (for Speech → ISL)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Karthik-Ramkumar/IsharaAI.git
   cd IsharaAI
   ```

2. **Set up a Virtual Environment** (Highly Recommended):
   ```bash
   # Windows
   py -3.9 -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Models**:
   Ensure the following models are in the `models/` directory:
   - `model.h5` (Gesture classifier)
   - `hand_landmarker.task` (MediaPipe model)
   - `vosk-model-small-en-us-0.15` (Speech model)

## Running the App
```bash
python app.py
```

## How to Use
- **ISL to Speech**: Show hand signs to the camera. Hold a sign steady for 0.5s to confirm. Use **Spacebar** on your keyboard to add spaces between words. Click **Speak Word** to hear the audio.
- **Text to ISL**: Type text in the box and press Enter or "Translate".
- **Speech to ISL**: Click "Record", speak for up to 5 seconds, and watch the ISL signs play back.
