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

# Load model
print("Loading ISL model...")
model = keras.models.load_model("model.h5")
print("✓ Model loaded!\n")

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# Sentence building variables
current_sentence = ""
last_detected = ""
frame_count = 0
stability_threshold = 10  # Frames to confirm detection

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

def draw_text_box(image, sentence, instructions):
    """Draw sentence box and instructions"""
    h, w = image.shape[:2]
    
    # Draw sentence box (top)
    cv2.rectangle(image, (10, 10), (w-10, 80), (50, 50, 50), -1)
    cv2.rectangle(image, (10, 10), (w-10, 80), (0, 255, 0), 2)
    
    # Draw sentence
    display_text = sentence if sentence else "..."
    cv2.putText(image, display_text, (20, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
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

print("="*70)
print("INDIAN SIGN LANGUAGE DETECTOR - Sentence Builder")
print("="*70)
print("Detects: 1-9 and A-Z")
print("\nControls:")
print("  SPACE    - Add detected letter to sentence")
print("  ENTER    - Add space to sentence")
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
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            draw_landmarks(image, hand_landmarks)
            
            # Get landmark list
            landmark_list = calc_landmark_list(hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            
            # Predict
            predictions = model.predict(df, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            detected_label = alphabet[predicted_classes[0]]
            confidence = predictions[0][predicted_classes[0]]
            
            # Display current detection
            cv2.putText(image, f"Detected: {detected_label} ({confidence*100:.1f}%)", 
                       (image.shape[1]//2 - 200, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            break  # Only process first hand
    
    # Draw sentence box and instructions
    instructions = [
        "SPACE: Add letter | ENTER: Space | BACKSPACE: Delete",
        "C: Clear all | ESC: Exit",
        f"Current: {detected_label} ({confidence*100:.0f}%)" if detected_label else "Current: No hand"
    ]
    draw_text_box(image, current_sentence, instructions)
    
    cv2.imshow('ISL Sentence Builder', image)
    
    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE - add detected letter
        if detected_label and confidence > 0.5:
            current_sentence += detected_label
            print(f"Added '{detected_label}' → Sentence: {current_sentence}")
    elif key == 13:  # ENTER - add space
        current_sentence += " "
        print(f"Added space → Sentence: {current_sentence}")
    elif key == 8:  # BACKSPACE - delete last char
        if current_sentence:
            current_sentence = current_sentence[:-1]
            print(f"Deleted → Sentence: {current_sentence}")
    elif key == ord('c') or key == ord('C'):  # C - clear
        current_sentence = ""
        print("Sentence cleared!")

cap.release()
cv2.destroyAllWindows()
print("\n✓ Detector closed!")
if current_sentence:
    print(f"Final sentence: {current_sentence}")
