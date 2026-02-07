# Building IsharaAI Executable

## Quick Build (Local)

### Method 1: Using the build script

```bash
python build_exe.py
```

The executable will be created in the `dist` folder.

### Method 2: Manual PyInstaller

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller --name=IsharaAI \
    --onefile \
    --windowed \
    --add-data="data:data" \
    --add-data="models:models" \
    --add-data="src:src" \
    --add-data="config.py:." \
    --hidden-import=PIL \
    --hidden-import=cv2 \
    --hidden-import=mediapipe \
    --hidden-import=tensorflow \
    --hidden-import=textblob \
    --hidden-import=pyttsx3 \
    --collect-all=mediapipe \
    app.py
```

## Automated Build (GitHub Actions)

### Option 1: Create a Release Tag

```bash
# Tag your code
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will automatically:
- Build the Windows executable
- Create a ZIP package with models and data
- Create a GitHub Release with downloadable files

### Option 2: Manual Trigger

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Click on "Build and Release Windows Executable" workflow
4. Click "Run workflow"
5. Wait for the build to complete
6. Download the artifact from the workflow run

## Distribution

After building, you'll get:
- `IsharaAI.exe` - The standalone executable
- Required folders:
  - `models/` - ML models and hand detector
  - `data/` - ISL sign images

### Creating a distributable package:

```bash
# Create a folder
mkdir IsharaAI-Release

# Copy files
cp dist/IsharaAI.exe IsharaAI-Release/
cp -r models IsharaAI-Release/
cp -r data IsharaAI-Release/
cp README.md IsharaAI-Release/

# Create ZIP
zip -r IsharaAI-Windows.zip IsharaAI-Release/
```

## Download Links

### GitHub Releases (Automated Builds)

Once you create a release tag, the executable will be available at:
```
https://github.com/Karthik-Ramkumar/IsharaAI/releases/latest
```

Direct download link for latest version:
```
https://github.com/Karthik-Ramkumar/IsharaAI/releases/latest/download/IsharaAI-Windows.zip
```

### For Users

Share this link with users to download the application:
```
https://github.com/Karthik-Ramkumar/IsharaAI/releases/latest
```

Or create a custom short link using services like:
- bit.ly
- tinyurl.com
- GitHub's own release page

## Building for Different Platforms

### Windows
```bash
# Already configured above
python build_exe.py
```

### macOS
```bash
pyinstaller --name=IsharaAI \
    --onefile \
    --windowed \
    --add-data="data:data" \
    --add-data="models:models" \
    --add-data="src:src" \
    app.py
```

### Linux
```bash
pyinstaller --name=IsharaAI \
    --onefile \
    --add-data="data:data" \
    --add-data="models:models" \
    --add-data="src:src" \
    app.py
```

## Troubleshooting

### Issue: "File is too large"

Some model files are large. You might need to use Git LFS:

```bash
git lfs install
git lfs track "models/*.task"
git lfs track "models/*.h5"
git lfs track "models/*.onnx"
```

### Issue: "ModuleNotFoundError"

Add the missing module to `--hidden-import` in the build command.

### Issue: "Antivirus blocks the .exe"

This is normal for PyInstaller executables. Users can:
1. Add an exception in their antivirus
2. Run the Python version directly
3. Wait for Windows SmartScreen to recognize the app

### Issue: "Models not found"

Make sure the `models` and `data` folders are in the same directory as the `.exe` file.

## File Size Optimization

To reduce executable size:

1. **Remove unused models**:
   ```bash
   # Only include the models you need
   --add-data="models/model.h5:models"
   --add-data="models/hand_landmarker.task:models"
   ```

2. **Compress with UPX**:
   ```bash
   pip install pyinstaller[upx]
   pyinstaller --upx-dir=/path/to/upx ...
   ```

3. **Exclude unnecessary packages**:
   ```bash
   --exclude-module=matplotlib
   --exclude-module=scipy
   ```

## Publishing to GitHub Releases

1. Create a git tag:
   ```bash
   git tag -a v1.0.0 -m "Initial release"
   git push origin v1.0.0
   ```

2. The GitHub Action will automatically build and create a release

3. Users can download from:
   ```
   https://github.com/Karthik-Ramkumar/IsharaAI/releases/latest
   ```

## Creating a Direct Download Link

After the GitHub Action creates a release, the download link will be:

```
https://github.com/Karthik-Ramkumar/IsharaAI/releases/latest/download/IsharaAI-Windows.zip
```

You can shorten this with bit.ly or create a custom domain redirect.
