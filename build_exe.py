"""
Build script to create executable for IsharaAI application.
This will create a standalone .exe file that can be distributed.
"""
import os
import subprocess
import sys

def build_executable():
    """Build the executable using PyInstaller."""
    
    # Install PyInstaller if not already installed
    print("Installing PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # PyInstaller command
    pyinstaller_cmd = [
        "pyinstaller",
        "--name=IsharaAI",
        "--onefile",  # Create a single executable
        "--windowed",  # No console window (GUI app)
        "--icon=data/raw/isl_images/A/1.jpg",  # Optional: add an icon
        "--add-data=data:data",  # Include data folder
        "--add-data=models:models",  # Include models folder
        "--add-data=src:src",  # Include src folder
        "--add-data=config.py:.",  # Include config file
        "--hidden-import=PIL",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=cv2",
        "--hidden-import=mediapipe",
        "--hidden-import=tensorflow",
        "--hidden-import=textblob",
        "--hidden-import=pyttsx3",
        "--hidden-import=sounddevice",
        "--hidden-import=vosk",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--collect-all=mediapipe",
        "--collect-all=tensorflow",
        "--collect-all=textblob",
        "app.py"
    ]
    
    print("Building executable...")
    print(" ".join(pyinstaller_cmd))
    
    try:
        subprocess.run(pyinstaller_cmd, check=True)
        print("\n✅ Build successful!")
        print(f"Executable created at: {os.path.join(script_dir, 'dist', 'IsharaAI.exe')}")
        print("\nYou can now distribute the 'dist' folder or just the IsharaAI.exe file.")
        print("Note: Users will also need the 'models' and 'data' folders in the same directory.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_executable()
