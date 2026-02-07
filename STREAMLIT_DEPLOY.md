# IsharaAI - Streamlit Web App Deployment Guide

## 📋 Overview
This guide explains how to deploy IsharaAI as a web application using Streamlit Cloud.

## 🚀 Quick Start (Local Testing)

### 1. Install Dependencies
```bash
cd /home/karthik/IsharaAI
pip install -r requirements-streamlit.txt
```

### 2. Run Locally
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1: Prepare Repository
1. **Commit all files to GitHub:**
   ```bash
   git add streamlit_app.py requirements-streamlit.txt .streamlit/
   git commit -m "Add Streamlit web app"
   git push origin main
   ```

2. **Ensure these files exist in your repo:**
   - ✅ `streamlit_app.py` (main app)
   - ✅ `requirements-streamlit.txt` (dependencies)
   - ✅ `.streamlit/config.toml` (theme config)
   - ✅ `IsharaAI/models/model.h5` (ISL model)
   - ✅ `IsharaAI/models/hand_landmarker.task` (MediaPipe model)
   - ✅ `IsharaAI/train_img/` (sign images folder)

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Click "Sign up" or "Sign in with GitHub"

2. **Create New App:**
   - Click "New app" button
   - Select your repository: `Karthik-Ramkumar/IsharaAI`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Wait for Deployment:**
   - Streamlit will install dependencies
   - Build time: ~5-10 minutes (first time)
   - You'll get a URL like: `https://isharaai.streamlit.app`

### Step 3: Configure Advanced Settings (Optional)

In Streamlit Cloud dashboard:
- **Python version:** 3.11 or 3.12
- **Secrets:** Add any API keys if needed (not required for basic app)

## 🎯 Key Differences from Desktop App

| Feature | Desktop (Tkinter) | Web (Streamlit) |
|---------|------------------|-----------------|
| **Camera** | Live video feed | Snapshot-based (st.camera_input) |
| **Performance** | 60fps real-time | ~1-2fps (manual capture) |
| **TTS** | pyttsx3/Piper | Browser-based (limited) |
| **Autocorrect** | TextBlob | Can add TextBlob |
| **Installation** | Requires Python | Just open URL |
| **Deployment** | Desktop only | Accessible anywhere |

## ⚠️ Limitations

1. **Camera Mode:**
   - Browser security requires manual photo capture
   - Not continuous real-time like desktop app
   - User must click "Take photo" for each sign

2. **Model Size:**
   - Streamlit Cloud has 1GB RAM limit (free tier)
   - TensorFlow model (11MB) + MediaPipe should work
   - If issues, consider model quantization

3. **Performance:**
   - Slower than desktop due to server-side processing
   - Each prediction requires upload → process → download

4. **Speech Recognition:**
   - Desktop uses Vosk (offline)
   - Web requires browser APIs (limited support)

## 🔧 Troubleshooting

### Error: "File not found: model.h5"
**Fix:** Ensure models are committed to git:
```bash
git lfs install
git lfs track "*.h5" "*.task"
git add .gitattributes IsharaAI/models/
git commit -m "Add models with LFS"
git push origin main
```

### Error: "Memory limit exceeded"
**Fix:** Streamlit free tier has 1GB RAM. Options:
1. Upgrade to Streamlit Cloud paid tier
2. Use model quantization
3. Deploy to Railway.app or Render.com (more RAM)

### Camera not working
**Fix:** Browser security requires HTTPS. Streamlit Cloud provides this automatically.

### Slow predictions
**Fix:** This is expected on web. For better performance:
1. Keep desktop app for heavy use
2. Use web app for demos/accessibility
3. Consider paid hosting for better resources

## 🌐 Alternative Deployment Options

### Option 1: Railway.app (More RAM)
```bash
# Free tier: 512MB → 8GB RAM
# Visit: https://railway.app
# Connect GitHub → Deploy
```

### Option 2: Render.com
```bash
# Free tier: 512MB RAM
# Visit: https://render.com
# New Web Service → Connect GitHub
```

### Option 3: Hugging Face Spaces
```bash
# Free GPU access
# Visit: https://huggingface.co/spaces
# Create new Space → Streamlit SDK
```

## 📊 Recommended Setup

**For Best Experience:**
- **Desktop App:** Heavy daily use, best performance, offline
- **Web App:** Demos, sharing, accessibility, mobile access

**Deployment Strategy:**
1. Keep desktop app as primary (better performance)
2. Deploy web app for:
   - Public demos
   - Mobile users
   - Quick access without installation
   - Portfolio showcase

## 🎓 Next Steps

1. **Test locally first:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Optimize if needed:**
   - Reduce model size
   - Add caching with @st.cache_resource
   - Implement session state management

3. **Deploy to Streamlit Cloud** (easiest)

4. **Share your URL:**
   - Get link from Streamlit Cloud dashboard
   - Share on social media, resume, portfolio

## 📞 Support

- **Streamlit Docs:** https://docs.streamlit.io
- **Community Forum:** https://discuss.streamlit.io
- **GitHub Issues:** https://github.com/Karthik-Ramkumar/IsharaAI/issues

---

**Created by:** Karthik Ramkumar  
**Repository:** https://github.com/Karthik-Ramkumar/IsharaAI  
**License:** MIT
