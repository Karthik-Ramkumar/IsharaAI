# IsharaAI - Indian Sign Language Translation System

An AI-based real-time translation system for Indian Sign Language (ISL) that enables bidirectional communication between hearing/speech-impaired individuals and the general population.

## 🎯 Features

- **ISL to Text**: Real-time recognition of ISL gestures and conversion to text
- **ISL to Speech**: Real-time recognition of ISL gestures with text-to-speech output
- **Deep Learning Model**: CNN-based gesture recognition with high accuracy
- **Webcam Integration**: Works with standard webcams on low-cost devices
- **Open Source**: Built entirely with free, open-source Python libraries

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Kaggle account (for dataset download)

### Installation

1. Clone the repository:
```bash
cd IsharaAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API (for dataset download):
   - Create a Kaggle account at https://www.kaggle.com
   - Go to Account Settings → API → Create New API Token
   - Place the downloaded `kaggle.json` file in `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Usage

#### Method 1: Using the Main Menu

```bash
python main.py
```

This will show an interactive menu with the following options:
1. Download ISL Dataset from Kaggle
2. Train ISL Recognition Model
3. Run ISL to Text (Webcam)
4. Run ISL to Speech (Webcam)

#### Method 2: Running Individual Scripts

**Step 1: Download Dataset**
```bash
python download_dataset.py
```

**Step 2: Train Model**
```bash
python train_model.py
```
This will train the CNN model on the ISL dataset. The trained model will be saved in the `models/` directory.

**Step 3: Run ISL to Text**
```bash
python isl_to_text.py
```
- Show ISL gestures to your webcam
- Press **SPACE** to add recognized gesture to text
- Press **C** to clear text
- Press **Q** to quit

**Step 4: Run ISL to Speech**
```bash
python isl_to_speech.py
```
- Show ISL gestures to your webcam
- Press **SPACE** to add gesture to sentence
- Press **ENTER** to speak the complete sentence
- Press **S** to speak the last recognized gesture immediately
- Press **C** to clear text
- Press **Q** to quit

## 📊 Dataset

This project uses the [Indian Sign Language Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) from Kaggle, which contains images of ISL hand gestures for various letters and words.

## 🏗️ Project Structure

```
IsharaAI/
├── config.py              # Configuration settings
├── download_dataset.py    # Dataset download script
├── train_model.py         # Model training script
├── isl_to_text.py        # ISL to text converter
├── isl_to_speech.py      # ISL to speech converter
├── main.py               # Main runner script
├── requirements.txt      # Python dependencies
├── data/                 # Dataset directory
├── models/               # Trained models directory
└── README.md            # This file
```

## 🔧 Configuration

Edit `config.py` to customize:
- Image size for training
- Batch size and epochs
- Learning rate
- Confidence threshold for predictions
- Webcam settings

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework for CNN model
- **OpenCV**: Real-time computer vision and webcam capture
- **MediaPipe**: Hand landmark detection (optional enhancement)
- **pyttsx3**: Text-to-speech conversion
- **NumPy/Pandas**: Data processing
- **scikit-learn**: Dataset splitting and preprocessing

## 📈 Model Architecture

The system uses a Convolutional Neural Network (CNN) with:
- 4 convolutional blocks with batch normalization and dropout
- MaxPooling layers for dimension reduction
- Dense layers for classification
- Softmax activation for multi-class prediction

## 🎯 Performance Optimization

The model includes several optimization techniques:
- Data augmentation (rotation, flip, zoom)
- Batch normalization for stable training
- Dropout layers to prevent overfitting
- Early stopping to avoid overtraining
- Learning rate reduction on plateau

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📝 Future Enhancements

- [ ] Speech to ISL (reverse translation)
- [ ] Support for ISL sentences and grammar
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Continuous gesture recognition (no button press needed)
- [ ] MediaPipe integration for improved hand tracking
- [ ] Context-aware predictions

## 📄 License

This project is open source and available for educational and non-commercial use.

## 🙏 Acknowledgments

- Indian Sign Language Dataset by Prathu Marikeri on Kaggle
- TensorFlow and OpenCV communities
- All contributors to open-source ML/CV libraries

## 📧 Contact

For questions or support, please open an issue on the repository.

---

**Note**: This system is designed for educational and accessibility purposes. For production use, consider additional training data and fine-tuning for your specific use case.
Real-Time Indian Sign Language and Speech Translation System
