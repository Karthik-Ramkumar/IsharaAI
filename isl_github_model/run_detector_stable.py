#!/usr/bin/env python3
"""ISL Detection using MediaPipe 0.10.32 Tasks API"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import threading

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

def speak_text(text):
    """Speak text in a separate thread to not block video"""
    def speak():
        if text.strip():
            tts_engine.say(text)
            tts_engine.runAndWait()
    
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

# Load model
print("Loading ISL model...")
model = keras.models.load_model("model.h5")
print("✓ Model loaded!\n")

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Sentence building variables
current_sentence = ""
last_detected = ""
detection_history = []  # Track last N detections for stability
stability_frames = 6  # Number of consistent frames needed (reduced from 8)
confidence_threshold = 0.60  # Minimum confidence to consider (reduced from 0.70)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),  # Thumb
    (0,5), (5,6), (6,7), (7,8),  # Index
    (0,9), (9,10), (10,11), (11,12),  # Middle
    (0,13), (13,14), (14,15), (15,16),  # Ring
    (0,17), (17,18), (18,19), (19,20),  # Pinky
    (5,9), (9,13), (13,17)  # Palm
]

def draw_landmarks(image, landmarks):
    """Draw hand landmarks on image"""
    h, w = image.shape[:2]
    
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    # Draw landmark points
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(image, (x, y), 3, (0, 255, 255), -1)

def calc_landmark_list(landmarks_list):
    """Convert normalized landmarks to list"""
    landmark_point = []
    for landmark in landmarks_list:
        landmark_point.append([landmark.x, landmark.y])
    return landmark_point

def pre_process_landmark(landmark_list):
    """Preprocess landmarks"""
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else n
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

def draw_text_box(image, sentence, instructions, top_predictions=None):
    """Draw sentence box, instructions, and top predictions"""
    h, w = image.shape[:2]
    
    # Draw sentence box (top)
    cv2.rectangle(image, (10, 10), (w-10, 80), (50, 50, 50), -1)
    cv2.rectangle(image, (10, 10), (w-10, 80), (0, 255, 0), 2)
    
    # Draw sentence
    display_text = sentence if sentence else "..."
    cv2.putText(image, display_text, (20, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Draw top predictions box (right side)
    if top_predictions:
        pred_x = w - 350
        cv2.rectangle(image, (pred_x, 100), (w-10, 340), (50, 50, 50), -1)
        cv2.rectangle(image, (pred_x, 100), (w-10, 340), (255, 200, 0), 2)
        
        cv2.putText(image, "Top Predictions:", (pred_x+10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, (label, conf) in enumerate(top_predictions):
            y_pos = 170 + i * 50
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(image, f"{i+1}. {label}: {conf:.1f}%", 
                       (pred_x+10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Draw instructions box (bottom)
    cv2.rectangle(image, (10, h-120), (w-10, h-10), (50, 50, 50), -1)
    cv2.rectangle(image, (10, h-120), (w-10, h-10), (0, 200, 255), 2)
    
    # Instructions
    cv2.putText(image, instructions[0], (20, h-90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, instructions[1], (20, h-60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, instructions[2], (20, h-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def get_stable_prediction(current_pred, history, required_frames):
    """Get prediction only if it's been stable for N frames"""
    history.append(current_pred)
    if len(history) > required_frames:
        history.pop(0)
    
    if len(history) < required_frames:
        return None
    
    # Check if all recent predictions are the same
    if all(p == history[-1] for p in history[-required_frames:]):
        return history[-1]
    return None

print("="*70)
print("INDIAN SIGN LANGUAGE TO SPEECH - STABLE VERSION")
print("="*70)
print("Detects: 1-9 and A-Z")
print(f"Stability: {stability_frames} frames | Confidence: {confidence_threshold*100:.0f}%")
print("\nControls:")
print("  SPACE    - Add detected letter to sentence")
print("  ENTER    - Add space to sentence")
print("  S        - Speak current sentence (Text-to-Speech)")
print("  BACKSPACE- Delete last character")
print("  C        - Clear sentence")
print("  ESC      - Exit")
print("="*70)
print()

# Setup hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    
    if not success:
        continue
    
    # Convert to RGB and detect
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    
    detected_label = ""
    confidence = 0.0
    stable_label = None
    top_predictions = None
    best_hand_predictions = None
    best_confidence = 0.0
    
    if detection_result.hand_landmarks:
        all_hand_results = []
        
        # Process ALL detected hands
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Draw landmarks for all hands
            draw_landmarks(image, hand_landmarks)
            
            # Get landmark list
            landmark_list = calc_landmark_list(hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            
            # Predict
            predictions = model.predict(df, verbose=0)[0]
            
            # Get top prediction for this hand
            best_idx = np.argmax(predictions)
            hand_label = alphabet[best_idx]
            hand_confidence = predictions[best_idx]
            
            all_hand_results.append({
                'label': hand_label,
                'confidence': hand_confidence,
                'predictions': predictions,
                'hand_idx': idx
            })
        
        # Choose the hand with highest confidence
        if all_hand_results:
            best_result = max(all_hand_results, key=lambda x: x['confidence'])
            detected_label = best_result['label']
            confidence = best_result['confidence']
            predictions = best_result['predictions']
            
            # Get top 3 predictions from best hand
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [(alphabet[i], predictions[i]*100) for i in top_3_indices]
            
            # Show which hand is being used
            hand_text = f"Hand {best_result['hand_idx']+1}" if len(all_hand_results) > 1 else "Hand"
            
            # Only consider if confidence is high enough
            if confidence >= confidence_threshold:
                stable_label = get_stable_prediction(detected_label, detection_history, stability_frames)
                
                if stable_label:
                    # Show stable detection in green
                    cv2.putText(image, f"STABLE: {stable_label} ({confidence*100:.1f}%) [{hand_text}]", 
                               (image.shape[1]//2 - 300, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
                else:
                    # Show unstable detection in yellow
                    cv2.putText(image, f"Detecting: {detected_label} ({confidence*100:.1f}%) [{hand_text}]", 
                               (image.shape[1]//2 - 300, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            else:
                detection_history.clear()
                cv2.putText(image, f"Low confidence: {detected_label} ({confidence*100:.1f}%)", 
                           (image.shape[1]//2 - 250, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    else:
        detection_history.clear()
    
    # Draw sentence box, instructions, and top predictions
    instructions = [
        "SPACE: Add letter | ENTER: Space | S: Speak | BACKSPACE: Delete",
        "C: Clear all | ESC: Exit",
        f"Stable: {stable_label if stable_label else 'Waiting...'} | Frames: {len(detection_history)}/{stability_frames}"
    ]
    draw_text_box(image, current_sentence, instructions, top_predictions)
    
    cv2.imshow('ISL Sentence Builder', image)
    
    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE - add detected letter (only if stable)
        if stable_label:
            current_sentence += stable_label
            print(f"Added '{stable_label}' → Sentence: {current_sentence}")
            detection_history.clear()  # Reset after adding
    elif key == 13:  # ENTER - add space
        current_sentence += " "
        print(f"Added space → Sentence: {current_sentence}")
    elif key == ord('s') or key == ord('S'):  # S - speak sentence
        if current_sentence.strip():
            print(f"🔊 Speaking: {current_sentence}")
            speak_text(current_sentence)
        else:
            print("⚠ No text to speak!")
    elif key == 8:  # BACKSPACE - delete last char
        if current_sentence:
            current_sentence = current_sentence[:-1]
            print(f"Deleted → Sentence: {current_sentence}")
    elif key == ord('c') or key == ord('C'):  # C - clear
        current_sentence = ""
        detection_history.clear()
        print("Sentence cleared!")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Detector closed!")
if current_sentence:
    print(f"Final sentence: {current_sentence}")
