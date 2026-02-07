"""
Script: test_camera.py
Description: Standalone test for ISL detection using trained Keras model
"""
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import copy
import itertools
import string
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras
import pandas as pd

# alphabet
ISL_ALPHABET = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    
    # Flatten
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

def main():
    print("Initializing...")
    
    # Load Keras model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    try:
        model = keras.models.load_model(model_path)
        print("✓ Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Hand Landmarker
    task_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")
    if not os.path.exists(task_path):
        print(f"Error: Task not found at {task_path}")
        return
        
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("✓ Detector initialized")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    print("Camera opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw landmarks
                for lm in hand_landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Predict
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                processed = pre_process_landmark(landmark_list)
                
                df = pd.DataFrame(processed).transpose()
                predictions = model.predict(df, verbose=0)
                predicted_index = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)
                
                label = ISL_ALPHABET[predicted_index]
                cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('ISL Detection Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
