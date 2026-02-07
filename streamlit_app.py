import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tensorflow as tf
from pathlib import Path
import time
import itertools

# Page config
st.set_page_config(
    page_title="IsharaAI - ISL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
    }
    .stButton>button {
        background-color: #06B6D4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0891B2;
    }
    h1, h2, h3 {
        color: #06B6D4;
    }
    .output-box {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #06B6D4;
        min-height: 150px;
        color: white;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_time' not in st.session_state:
    st.session_state.prediction_time = 0

# Load models
@st.cache_resource
def load_models():
    """Load TensorFlow model and MediaPipe hand detector"""
    # Load ISL model - adjust path for both streamlit_app.py and when run from IsharaAI/
    current_dir = Path(__file__).parent
    
    # Try different possible paths
    possible_model_paths = [
        current_dir / "IsharaAI" / "models" / "model.h5",
        current_dir / "models" / "model.h5",
    ]
    
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        raise FileNotFoundError(f"model.h5 not found. Tried: {possible_model_paths}")
    
    model = tf.keras.models.load_model(str(model_path))
    
    # Load MediaPipe HandLandmarker
    possible_task_paths = [
        current_dir / "IsharaAI" / "models" / "hand_landmarker.task",
        current_dir / "models" / "hand_landmarker.task",
    ]
    
    task_path = None
    for path in possible_task_paths:
        if path.exists():
            task_path = path
            break
    
    if not task_path:
        raise FileNotFoundError(f"hand_landmarker.task not found. Tried: {possible_task_paths}")
    
    base_options = mp.tasks.BaseOptions(model_asset_path=str(task_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    
    return model, detector

# Sign classes (same order as desktop app: numbers first, then letters)
SIGN_CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def process_landmarks(landmarks, image_width, image_height):
    """Convert MediaPipe landmarks to model input (same as desktop app)"""
    if not landmarks or len(landmarks) == 0:
        return None
    
    # Use first hand
    hand_landmarks = landmarks[0]
    
    # Step 1: Convert to pixel coordinates
    landmark_point = []
    for landmark in hand_landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    # Step 2: Make relative to wrist (landmark 0)
    base_x, base_y = landmark_point[0][0], landmark_point[0][1]
    for i in range(len(landmark_point)):
        landmark_point[i][0] -= base_x
        landmark_point[i][1] -= base_y
    
    # Step 3: Flatten to 1D list
    import itertools
    flat_landmarks = list(itertools.chain.from_iterable(landmark_point))
    
    # Step 4: Normalize
    max_value = max(list(map(abs, flat_landmarks)))
    if max_value != 0:
        flat_landmarks = [n / max_value for n in flat_landmarks]
    
    return flat_landmarks

def predict_sign(image, model, detector):
    """Predict ISL sign from image"""
    # Convert PIL to RGB array
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # Convert to MediaPipe format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    
    # Detect hands
    result = detector.detect(mp_image)
    
    if not result.hand_landmarks:
        return None, 0.0
    
    # Process landmarks (same as desktop app)
    features = process_landmarks(result.hand_landmarks, img_w, img_h)
    if features is None:
        return None, 0.0
    
    # Convert to DataFrame (same as desktop app)
    import pandas as pd
    df = pd.DataFrame([features])
    
    # Predict
    prediction = model.predict(df, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    if confidence > 0.65:
        return SIGN_CLASSES[predicted_class], confidence
    
    return None, confidence

# Main app
def main():
    st.title("🤟 IsharaAI - Indian Sign Language Translator")
    st.markdown("### Bridging Communication Gaps with AI")
    
    # Load models
    try:
        model, detector = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure model.h5 and hand_landmarker.task are in IsharaAI/models/")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📷 ISL to Text", "✍️ Text to ISL", "🎤 Speech to ISL"])
    
    # Tab 1: ISL to Text (Live Camera)
    with tab1:
        st.header("ISL to Text Translation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Camera Feed")
            
            # Check if camera is available (won't work on Streamlit Cloud)
            st.info("ℹ️ **Note:** Live camera only works locally. On Streamlit Cloud, use snapshot mode below or download the desktop app.")
            
            # Camera controls
            run_camera = st.checkbox("Start Live Camera (Local Only)", value=False, key="camera_running")
            
            # Placeholder for camera feed
            camera_placeholder = st.empty()
            status_placeholder = st.empty()
            
            if run_camera:
                import cv2
                
                # Open camera
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                frame_count = 0
                
                try:
                    while run_camera:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.error("Failed to access camera")
                            break
                        
                        # Mirror the frame
                        frame = cv2.flip(frame, 1)
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process every 3rd frame for detection
                        if frame_count % 3 == 0:
                            # Create PIL image for prediction
                            pil_image = Image.fromarray(frame_rgb)
                            sign, confidence = predict_sign(pil_image, model, detector)
                            
                            if sign and confidence > 0.65:
                                # Add to recognized text
                                current_time = time.time()
                                if (st.session_state.last_prediction != sign or 
                                    current_time - st.session_state.prediction_time > 1.5):
                                    st.session_state.recognized_text += sign
                                    st.session_state.last_prediction = sign
                                    st.session_state.prediction_time = current_time
                                
                                # Draw detection on frame
                                cv2.putText(frame_rgb, f"{sign} ({confidence:.0%})", 
                                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                          1.2, (6, 182, 212), 3)
                                status_placeholder.success(f"Detected: **{sign}** ({confidence:.1%})")
                            else:
                                status_placeholder.info("Show a sign...")
                        
                        # Display frame
                        camera_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                        
                        frame_count += 1
                        time.sleep(0.033)  # ~30 fps
                        
                except Exception as e:
                    status_placeholder.error(f"Camera error: {e}")
                finally:
                    cap.release()
            else:
                camera_placeholder.info("👆 Check the box above to start the camera")
        
        with col2:
            st.subheader("Recognized Text")
            
            # Display recognized text
            st.markdown(f"""
            <div class="output-box">
                {st.session_state.recognized_text or "Start signing to see text here..."}
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Add Space"):
                    st.session_state.recognized_text += " "
                    st.rerun()
            with col_b:
                if st.button("Backspace"):
                    st.session_state.recognized_text = st.session_state.recognized_text[:-1]
                    st.rerun()
            with col_c:
                if st.button("Clear All"):
                    st.session_state.recognized_text = ""
                    st.session_state.last_prediction = None
                    st.rerun()
    
    # Tab 2: Text to ISL
    with tab2:
        st.header("Text to ISL Translation")
        
        text_input = st.text_input("Enter text to translate:", placeholder="Type something...")
        
        if text_input:
            signs = [char.upper() for char in text_input if char.upper() in SIGN_CLASSES]
            
            if signs:
                st.subheader("Sign Language Images")
                
                # Display signs in rows
                cols_per_row = 6
                for i in range(0, len(signs), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(signs):
                            sign = signs[i + j]
                            
                            # Try multiple possible paths
                            current_dir = Path(__file__).parent
                            possible_paths = [
                                current_dir / "IsharaAI" / "train_img" / sign / f"{sign}1.jpg",
                                current_dir / "train_img" / sign / f"{sign}1.jpg",
                                current_dir / "IsharaAI" / "data" / "train" / sign / f"{sign}1.jpg",
                            ]
                            
                            sign_path = None
                            for path in possible_paths:
                                if path.exists():
                                    sign_path = path
                                    break
                            
                            if sign_path:
                                with col:
                                    st.image(str(sign_path), caption=sign, width="stretch")
                            else:
                                with col:
                                    # Create placeholder
                                    st.markdown(f"""
                                    <div style="background: #1E293B; padding: 20px; text-align: center; 
                                                border: 2px solid #06B6D4; border-radius: 8px; min-height: 100px;">
                                        <h2 style="color: #06B6D4; margin: 0;">{sign}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
            else:
                st.warning("No supported characters found. Use A-Z or 1-9.")
    
    # Tab 3: Speech to ISL
    with tab3:
        st.header("Speech to ISL Translation")
        st.info("🎤 Speech recognition is browser-based in Streamlit")
        
        # Use Streamlit's audio input (experimental)
        st.markdown("""
        **How to use:**
        1. Click the microphone button in your browser
        2. Grant microphone permissions
        3. Speak clearly
        4. The text will appear below
        """)
        
        # Text input as fallback
        speech_text = st.text_area("Or type your speech here:", placeholder="Enter text...")
        
        if speech_text:
            signs = [char.upper() for char in speech_text if char.upper() in SIGN_CLASSES]
            
            if signs:
                st.subheader("Sign Language Images")
                
                cols_per_row = 6
                for i in range(0, len(signs), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(signs):
                            sign = signs[i + j]
                            
                            # Try multiple possible paths
                            current_dir = Path(__file__).parent
                            possible_paths = [
                                current_dir / "IsharaAI" / "train_img" / sign / f"{sign}1.jpg",
                                current_dir / "train_img" / sign / f"{sign}1.jpg",
                                current_dir / "IsharaAI" / "data" / "train" / sign / f"{sign}1.jpg",
                            ]
                            
                            sign_path = None
                            for path in possible_paths:
                                if path.exists():
                                    sign_path = path
                                    break
                            
                            if sign_path:
                                with col:
                                    st.image(str(sign_path), caption=sign, width="stretch")
                            else:
                                with col:
                                    # Create placeholder
                                    st.markdown(f"""
                                    <div style="background: #1E293B; padding: 20px; text-align: center; 
                                                border: 2px solid #06B6D4; border-radius: 8px; min-height: 100px;">
                                        <h2 style="color: #06B6D4; margin: 0;">{sign}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About IsharaAI")
        st.markdown("""
        **Indian Sign Language Translator**
        
        - 🎯 35 Signs (A-Z, 1-9)
        - 🤖 TensorFlow + MediaPipe
        - 📊 28K+ Training Images
        - 🎓 65%+ Confidence Threshold
        
        **Features:**
        - Real-time ISL detection
        - Text to sign conversion
        - Speech to sign conversion
        """)
        
        st.divider()
        st.markdown("**GitHub:** [IsharaAI](https://github.com/Karthik-Ramkumar/IsharaAI)")

if __name__ == "__main__":
    main()
