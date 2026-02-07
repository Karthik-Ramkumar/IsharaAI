#!/usr/bin/env python3
"""
Download required model files for IsharaAI
Run this script after cloning the repository
"""
import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress"""
    print(f"Downloading {os.path.basename(destination)}...")
    urllib.request.urlretrieve(url, destination)
    print(f"✓ Downloaded to {destination}")

def main():
    print("="*70)
    print("IsharaAI Model Downloader")
    print("="*70)
    
    # Create directories
    models_dir = Path("models")
    isl_model_dir = Path("isl_github_model")
    models_dir.mkdir(exist_ok=True)
    isl_model_dir.mkdir(exist_ok=True)
    
    # Download MediaPipe hand landmarker (7.5MB)
    hand_landmarker_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    print("\n1. Downloading MediaPipe Hand Landmarker...")
    download_file(hand_landmarker_url, models_dir / "hand_landmarker.task")
    download_file(hand_landmarker_url, isl_model_dir / "hand_landmarker.task")
    
    # Download Vosk speech model (40MB)
    print("\n2. Downloading Vosk Speech Recognition Model...")
    vosk_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    vosk_zip = models_dir / "vosk-model.zip"
    
    if not (models_dir / "vosk-model-small-en-us-0.15").exists():
        download_file(vosk_url, vosk_zip)
        
        print("Extracting Vosk model...")
        with zipfile.ZipFile(vosk_zip, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        print("✓ Vosk model extracted")
        
        # Keep the zip file for reference
        print(f"✓ Vosk zip kept at {vosk_zip}")
    else:
        print("✓ Vosk model already exists")
    
    print("\n" + "="*70)
    print("MODEL FILES NEEDED (download separately):")
    print("="*70)
    print("\n⚠️  IMPORTANT: The following files are too large for GitHub:")
    print("\n1. models/model.h5 (11MB)")
    print("   - Pre-trained ISL gesture classifier")
    print("   - Download from: https://github.com/MaitreeVaria/Indian-Sign-Language-Detection")
    print("   - Or train your own using: python models/train_model.py")
    
    print("\n2. isl_github_model/model.h5 (11MB)")
    print("   - Same model, copy from models/model.h5")
    print("   - Or download from the GitHub repo above")
    
    print("\n3. isl_github_model/keypoint.csv (5.2MB)")
    print("   - Hand landmark training data")
    print("   - Download from: https://github.com/MaitreeVaria/Indian-Sign-Language-Detection")
    
    print("\n" + "="*70)
    print("QUICK SETUP:")
    print("="*70)
    print("\n# After downloading model.h5:")
    print("cp models/model.h5 isl_github_model/model.h5")
    
    print("\n" + "="*70)
    print("✓ Script completed!")
    print("="*70)

if __name__ == "__main__":
    main()
