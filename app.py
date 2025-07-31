import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Audio processing parameters (same as your model)
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 26
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Global variables for model and preprocessing
model = None
label_encoder = None
scaler = None
country_names = []

def load_model_and_preprocessors():
    """Load the trained model and preprocessing components"""
    global model, label_encoder, scaler, country_names
    
    try:
        # Try multiple model file extensions
        model_paths = [
            os.environ.get('MODEL_PATH', 'audio_country_enhanced_transformer_model.keras'),
            'audio_country_enhanced_transformer_model.keras',
            'audio_country_enhanced_transformer_model.h5',
            'model.keras',
            'model.h5'
        ]
        
        encoder_paths = [
            os.environ.get('ENCODER_PATH', 'label_encoder_enhanced_transformer.pkl'),
            'label_encoder_enhanced_transformer.pkl',
            'label_encoder.pkl',
            'encoder.pkl'
        ]
        
        scaler_paths = [
            os.environ.get('SCALER_PATH', 'feature_scaler_transformer.pkl'),
            'feature_scaler_transformer.pkl',
            'scaler.pkl'
        ]
        
        print("Loading model components...")
        
        # Load the trained model (try multiple formats)
        model = None
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path)
                    print(f"✅ Model loaded from: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {model_path}: {e}")
                    continue
        
        if model is None:
            print("⚠️  No model file found. Running in demo mode with mock predictions")
        
        # Load label encoder
        label_encoder = None
        for encoder_path in encoder_paths:
            if os.path.exists(encoder_path):
                try:
                    with open(encoder_path, 'rb') as f:
                        label_encoder = pickle.load(f)
                    country_names = label_encoder.classes_.tolist()
                    print(f"✅ Label encoder loaded with {len(country_names)} countries")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {encoder_path}: {e}")
                    continue
        
        if label_encoder is None:
            print("⚠️  No label encoder found. Using default country names")
            country_names = [
                "United States", "United Kingdom", "Canada", "Australia", 
                "Germany", "France", "Spain", "Italy", "Japan", "China",
                "India", "Brazil", "Mexico", "Russia", "South Africa"
            ]
        
        # Load feature scaler (optional)
        scaler = None
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    print(f"✅ Feature scaler loaded")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {scaler_path}: {e}")
                    continue
        
        if scaler is None:
            print("⚠️  No feature scaler found. Will use raw features")
        
        print(f"Loaded {len(country_names)} countries: {country_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

def load_and_preprocess_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_comprehensive_features(audio, sr=SAMPLE_RATE):
    """Extract comprehensive audio features (same as your model)"""
    features = []

    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend([mfcc, mfcc_delta, mfcc_delta2])

    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db[:26])

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    features.append(chroma)

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    features.extend([spectral_centroids, spectral_rolloff, spectral_bandwidth, zero_crossing_rate])

    # Harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_mfcc = librosa.feature.mfcc(y=harmonic, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    percussive_mfcc = librosa.feature.mfcc(y=percussive, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    features.extend([harmonic_mfcc, percussive_mfcc])

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP_LENGTH)
    features.append(spectral_contrast)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr, hop_length=HOP_LENGTH)
    features.append(tonnetz)

    # Combine all features
    combined_features = np.vstack(features)

    # Normalize features
    combined_features = (combined_features - np.mean(combined_features, axis=1, keepdims=True)) / (np.std(combined_features, axis=1, keepdims=True) + 1e-8)

    return combined_features.T

def predict_country(audio_file_path):
    """Predict country from audio file"""
    try:
        # Load and preprocess audio
        audio = load_and_preprocess_audio(audio_file_path)
        if audio is None:
            return None, "Error loading audio file"
        
        # Extract features
        features = extract_comprehensive_features(audio)
        
        # Use actual model if available, otherwise use mock predictions
        if model is not None:
            try:
                # Preprocess features (with or without scaler)
                features_reshaped = features.reshape(1, -1)
                
                if scaler is not None:
                    # Use scaler if available
                    features_scaled = scaler.transform(features_reshaped)
                    features_reshaped = features_scaled.reshape(1, features.shape[0], features.shape[1])
                    print("Using feature scaler")
                else:
                    # Use raw features if no scaler
                    features_reshaped = features.reshape(1, features.shape[0], features.shape[1])
                    print("Using raw features (no scaler)")
                
                # Make prediction
                predictions = model.predict(features_reshaped, verbose=0)
                predictions = predictions[0]  # Get first (and only) prediction
                
                print(f"Model prediction shape: {predictions.shape}")
                print(f"Predictions: {predictions}")
                
            except Exception as e:
                print(f"Error during model prediction: {e}")
                print("Falling back to mock predictions")
                predictions = np.random.dirichlet(np.ones(len(country_names)), 1)[0]
        else:
            # Mock predictions for demo
            print("Using mock predictions (model not loaded)")
            predictions = np.random.dirichlet(np.ones(len(country_names)), 1)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        results = []
        
        for i, idx in enumerate(top_indices):
            confidence = predictions[idx] * 100
            results.append({
                'country': country_names[idx],
                'confidence': round(confidence, 2),
                'rank': i + 1
            })
        
        return results, None
        
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

def create_audio_visualization(audio_file_path):
    """Create audio visualization plots"""
    try:
        audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Waveform
        axes[0, 0].plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1, 0])
        axes[1, 0].set_title('MFCC')
        fig.colorbar(img, ax=axes[1, 0])
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 1])
        axes[1, 1].set_title('Mel-spectrogram')
        fig.colorbar(img, ax=axes[1, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', countries=country_names)

@app.route('/predict', methods=['POST'])
def predict_audio():
    """Handle audio prediction for the game"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get predictions
        predictions, error = predict_country(filepath)
        if error:
            return jsonify({'error': error}), 500
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model on startup
    if load_model_and_preprocessors():
        print("Model loaded successfully!")
    else:
        print("Warning: Model not loaded. Running in demo mode.")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 