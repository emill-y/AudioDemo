<<<<<<< HEAD
# Transformer Audio Classifier Demo

A modern web application showcasing a Transformer-based audio classifier that can predict countries from audio samples with 97% accuracy.

## 🎯 Project Overview

This demo showcases our Brain Inspired ML Final Project - a Transformer Audio Classifier that successfully classifies audio samples of country names. After our initial LSTM model achieved only ~60% accuracy, we implemented a Transformer architecture that achieved 97% accuracy.

**Team:** Akhil, Chris, Eisha, Jason, & Lakshman

## ✨ Features

- **Interactive Audio Upload**: Upload WAV, MP3, M4A, or FLAC files (max 16MB)
- **Real-time Audio Recording**: Record audio directly in the browser
- **Comprehensive Audio Analysis**: Visualize waveform, spectrogram, MFCC, and mel-spectrogram
- **Advanced Predictions**: Get top 3 country predictions with confidence scores
- **Modern UI/UX**: Responsive design with smooth animations and intuitive interface
- **Performance Metrics**: Detailed model performance visualization
- **Technical Documentation**: Complete project overview and architecture details

## 🏗️ Architecture

### Model Architecture
- **Hybrid Transformer-CNN**: Combines transformer attention with CNN features
- **Multi-Head Attention**: 8-12 attention heads with positional encoding
- **Comprehensive Feature Extraction**: MFCC, Mel-spectrogram, Chroma, Spectral features
- **Advanced Data Augmentation**: 15x augmentation with time/frequency modifications

### Audio Processing Parameters
- **Sample Rate**: 22050 Hz
- **Duration**: 3 seconds
- **MFCC Coefficients**: 26
- **Mel Bands**: 128
- **FFT Window**: 2048
- **Hop Length**: 512

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd audio_classifier_demo
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
audio_classifier_demo/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Main page
│   ├── about.html        # About page
│   └── demo.html         # Demo page
├── static/               # Static assets
│   ├── css/
│   │   └── style.css     # Custom styles
│   ├── js/
│   │   └── main.js       # JavaScript utilities
│   └── images/           # Image assets
└── uploads/              # Temporary upload directory
```

## 🎨 Features in Detail

### 1. Audio Upload & Processing
- Drag-and-drop file upload
- Real-time file validation
- Progress indicators
- Multiple audio format support

### 2. Audio Visualization
- **Waveform**: Time-domain audio representation
- **Spectrogram**: Frequency-time visualization
- **MFCC**: Mel-frequency cepstral coefficients
- **Mel-spectrogram**: Mel-scale frequency analysis

### 3. Prediction System
- Top 3 country predictions
- Confidence scores with progress bars
- Historical prediction tracking
- Export capabilities

### 4. Interactive Demo
- Sample audio files for testing
- Real-time audio recording
- Performance comparison charts
- Model architecture visualization

## 🔧 Configuration

### Environment Variables
```bash
# Flask configuration
export FLASK_ENV=development
export FLASK_DEBUG=1
export SECRET_KEY=your-secret-key-here

# Model configuration
export MODEL_PATH=path/to/your/model.h5
export MAX_FILE_SIZE=16777216  # 16MB in bytes
```

### Model Integration
To integrate your actual trained model:

1. **Update model path** in `app.py`:
   ```python
   model = tf.keras.models.load_model('path/to/your/model.h5')
   ```

2. **Load preprocessing components**:
   ```python
   with open('path/to/label_encoder.pkl', 'rb') as f:
       label_encoder = pickle.load(f)
   
   with open('path/to/scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)
   ```

3. **Update country names** to match your training data

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

#### Option 1: Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

#### Option 2: Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

#### Option 3: Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

### Environment Setup
```bash
# Production environment
export FLASK_ENV=production
export FLASK_DEBUG=0
export SECRET_KEY=your-production-secret-key
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97% |
| **Training Time** | ~2 hours |
| **Inference Speed** | Fast |
| **Memory Usage** | Medium |
| **Supported Countries** | 15 |
| **Audio Duration** | 3 seconds |
| **Max File Size** | 16MB |

## 🔍 Technical Details

### Feature Extraction
- **MFCC**: 26 coefficients with delta and delta-delta
- **Mel-spectrogram**: 128 frequency bands
- **Chroma features**: 12 pitch classes
- **Spectral features**: Centroid, rolloff, bandwidth, zero-crossing rate
- **Harmonic/Percussive**: Separated components
- **Spectral contrast**: Timbral analysis
- **Tonnetz**: Harmonic analysis

### Data Augmentation
- **Time stretching**: 0.8x to 1.2x speed variations
- **Pitch shifting**: ±3 semitones
- **Noise addition**: Gaussian noise injection
- **Volume scaling**: 0.6x to 1.4x amplitude
- **Time shifting**: Random temporal shifts
- **Section reversal**: Partial audio reversal
- **Echo effects**: Delay-based augmentation

### Model Architecture
```
Input Audio → Feature Extraction → Transformer Blocks → Classification Head
     ↓              ↓                    ↓                ↓
  3s Audio    Comprehensive      Multi-Head        Softmax
              Features          Attention         Output
```

## 🛠️ Troubleshooting

### Common Issues

1. **Audio file not processing**
   - Check file format (WAV, MP3, M4A, FLAC)
   - Ensure file size < 16MB
   - Verify audio file integrity

2. **Model not loading**
   - Check model file path
   - Verify TensorFlow version compatibility
   - Ensure all dependencies installed

3. **Memory issues**
   - Reduce batch size in app.py
   - Increase server memory
   - Use smaller audio files

4. **Performance issues**
   - Enable GPU acceleration
   - Optimize model architecture
   - Use smaller feature dimensions

### Debug Mode
```bash
export FLASK_DEBUG=1
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Brain Inspired ML Final Project. All rights reserved.

## 🙏 Acknowledgments

- **Team Members**: Akhil, Chris, Eisha, Jason, & Lakshman
- **Course**: Brain Inspired Machine Learning
- **Technologies**: TensorFlow, Librosa, Flask, Bootstrap
- **Inspiration**: Transformer architecture for audio processing

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section

---

**Note**: This is a demo application. For production use, ensure proper security measures, error handling, and model optimization. 
=======
# AudioDemo
# AudioDemo
>>>>>>> e0eff3a567860619f924a061143cb932e33818ef
