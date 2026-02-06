#!/usr/bin/env python3
"""Diagnostic tool to test specific ISL letters"""
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
    
    # Draw landmark points with numbers
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        # Draw landmark number
        cv2.putText(image, str(idx), (x+8, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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

print("="*70)
print("ISL LETTER DIAGNOSTIC TOOL")
print("="*70)
print("This tool shows detailed predictions for each letter")
print("\nInstructions:")
print("1. Show a sign for I, T, or any problematic letter")
print("2. Check the TOP 10 predictions to see where your letter ranks")
print("3. If your letter is in top 10 but not #1, the sign might be ambiguous")
print("4. Landmarks are numbered 0-20 to help you adjust hand position")
print("\nPress ESC to exit")
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
    
    if detection_result.hand_landmarks:
        all_hand_results = []
        
        # Process ALL detected hands
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Draw landmarks with numbers
            draw_landmarks(image, hand_landmarks)
            
            # Get landmark list
            landmark_list = calc_landmark_list(hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            
            # Predict
            predictions = model.predict(df, verbose=0)[0]
            
            all_hand_results.append({
                'predictions': predictions,
                'hand_idx': idx
            })
        
        # Choose the hand with highest confidence
        if all_hand_results:
            best_result = max(all_hand_results, key=lambda x: np.max(x['predictions']))
            predictions = best_result['predictions']
            
            # Get TOP 10 predictions
            top_10_indices = np.argsort(predictions)[-10:][::-1]
            
            # Display top 10 on left side
            y_start = 50
            cv2.rectangle(image, (10, y_start-30), (350, y_start+370), (50, 50, 50), -1)
            cv2.rectangle(image, (10, y_start-30), (350, y_start+370), (255, 200, 0), 2)
            
            cv2.putText(image, "TOP 10 PREDICTIONS:", (20, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for i, idx in enumerate(top_10_indices):
                y_pos = y_start + 40 + i * 35
                label = alphabet[idx]
                conf = predictions[idx] * 100
                
                # Highlight I and T
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                if label in ['I', 'T']:
                    color = (0, 255, 255)  # Cyan for I and T
                
                text = f"{i+1}. {label}: {conf:.1f}%"
                cv2.putText(image, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show best prediction large
            best_label = alphabet[top_10_indices[0]]
            best_conf = predictions[top_10_indices[0]]
            
            cv2.putText(image, f"DETECTED: {best_label} ({best_conf*100:.1f}%)", 
                       (400, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            
            # Find where I and T rank
            i_rank = -1
            t_rank = -1
            try:
                i_idx = alphabet.index('I')
                t_idx = alphabet.index('T')
                
                all_sorted = np.argsort(predictions)[::-1]
                i_rank = int(np.where(all_sorted == i_idx)[0][0]) + 1
                t_rank = int(np.where(all_sorted == t_idx)[0][0]) + 1
                
                cv2.putText(image, f"Letter 'I' rank: #{i_rank} ({predictions[i_idx]*100:.1f}%)", 
                           (400, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(image, f"Letter 'T' rank: #{t_rank} ({predictions[t_idx]*100:.1f}%)", 
                           (400, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            except:
                pass
    else:
        cv2.putText(image, "No hand detected", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.imshow('ISL Diagnostic Tool', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Diagnostic tool closed!")
