# 🎤 Flag Guessing Game - Audio Country Classifier

A fun web application powered by a Transformer neural network that can classify country names from audio with 97% accuracy! Players see a flag emoji and speak the country name, then the AI tries to guess what they said.

## 🧠 About the Model

This project uses a sophisticated Transformer-based audio classifier developed as part of a Brain Inspired ML Final Project by **Team: Akhil, Chris, Eisha, Jason, & Lakshman**.

- **Model Type**: Hybrid CNN-Transformer Architecture
- **Accuracy**: 97% on country name classification
- **Features**: 120+ audio features including MFCC, mel-spectrograms, chroma, spectral features, and more
- **Training**: Enhanced with extensive data augmentation techniques

## 🎮 How to Play

1. **Look at the flag emoji** displayed on the screen
2. **Click "Start Recording"** and clearly say the country name
3. **Click "Submit Guess"** to let the AI analyze your audio
4. **See the results** - did the AI correctly identify what you said?
5. **Track your accuracy** and try to get the highest score possible!

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- Your trained model files:
  - `audio_country_enhanced_transformer_model.keras`
  - `label_encoder_enhanced_transformer.pkl`

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd flag-guessing-game
   pip install -r requirements.txt
   ```

2. **Add your model files**:
   ```bash
   # Copy your model files to the models directory
   cp /path/to/audio_country_enhanced_transformer_model.keras models/
   cp /path/to/label_encoder_enhanced_transformer.pkl models/
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   # Make sure your model files are in the models/ directory
   docker-compose up --build
   ```

2. **Access the application** at `http://localhost:5000`

### Cloud Deployment

#### Deploy to Heroku
```bash
# Install Heroku CLI first
heroku create your-app-name
git add .
git commit -m "Deploy flag guessing game"
git push heroku main
```

#### Deploy to Railway
```bash
# Connect your GitHub repo to Railway
# Add environment variables if needed
# Railway will automatically deploy from your main branch
```

#### Deploy to Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/flag-game
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/flag-game --platform managed
```

## 📁 Project Structure

```
flag-guessing-game/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── templates/
│   └── index.html        # Main game interface
├── static/
│   ├── css/
│   │   └── style.css     # Game styling
│   └── js/
│       └── app.js        # Game logic and audio handling
├── models/               # AI model files (you need to add these)
│   ├── audio_country_enhanced_transformer_model.keras
│   └── label_encoder_enhanced_transformer.pkl
└── README.md
```

## 🎯 Features

- **Real-time Audio Recording**: Browser-based microphone access
- **AI-Powered Classification**: 97% accurate transformer model
- **Interactive Game Interface**: Score tracking and round-based gameplay
- **Responsive Design**: Works on desktop and mobile devices
- **Flag Emoji Display**: Visual country representation
- **Detailed Results**: Shows confidence scores and top predictions
- **Modern UI**: Beautiful gradients and animations

## 🔧 Technical Details

### Audio Processing
- **Sample Rate**: 22,050 Hz
- **Duration**: 3 seconds max per recording
- **Features**: 120+ dimensional feature vectors including:
  - MFCC coefficients (26)
  - Mel-spectrograms (128 → 26)
  - Chroma features
  - Spectral centroids, rolloff, bandwidth
  - Zero crossing rate
  - Harmonic/percussive separation
  - Spectral contrast
  - Tonnetz features

### Model Architecture
- **Hybrid CNN-Transformer**: Combines convolutional and attention mechanisms
- **Positional Encoding**: For sequence understanding
- **Multi-Head Attention**: 12 attention heads
- **Feed-Forward Networks**: 768-dimensional hidden layers
- **Regularization**: Dropout, batch normalization, layer normalization

### Browser Compatibility
- **Chrome**: Full support
- **Firefox**: Full support  
- **Safari**: Full support
- **Edge**: Full support
- **Mobile browsers**: Supported with responsive design

## 🛠️ Customization

### Adding More Countries
1. Update the `COUNTRIES` dictionary in `app.py`
2. Retrain your model with additional country data
3. Update the label encoder

### Modifying Audio Parameters
Update the constants in `app.py`:
```python
SAMPLE_RATE = 22050  # Audio sample rate
DURATION = 3         # Max recording duration
N_MFCC = 26         # Number of MFCC features
N_MELS = 128        # Mel-spectrogram bands
```

## 🐛 Troubleshooting

### Model Loading Issues
- Ensure model files are in the `models/` directory
- Check file permissions
- Verify TensorFlow version compatibility

### Audio Recording Problems
- Check browser microphone permissions
- Use HTTPS for production (required for microphone access)
- Test with different browsers

### Performance Issues
- Consider using GPU for inference in production
- Adjust the number of workers in `Dockerfile`
- Implement model caching for faster predictions

## 📊 Performance Monitoring

The application includes a health check endpoint:
```bash
curl http://localhost:5000/health
```

Monitor key metrics:
- Model loading status
- Prediction latency
- Audio processing time
- Memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of an academic research project. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- **Brain Inspired ML Course**: For the theoretical foundation
- **TensorFlow/Keras**: For the deep learning framework
- **Librosa**: For audio processing capabilities
- **Flask**: For the web framework
- **Team Members**: Akhil, Chris, Eisha, Jason, & Lakshman

---

**🎵 Ready to test your pronunciation against AI? Let the games begin!** 🎵
