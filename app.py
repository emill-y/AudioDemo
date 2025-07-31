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
        # Load the model (you'll need to provide the actual model file)
        # model = tf.keras.models.load_model('path_to_your_model.h5')
        
        # For demo purposes, we'll create a mock model
        # In production, you'd load your actual trained model
        print("Loading model components...")
        
        # Mock country names for demo
        country_names = [
            "United States", "United Kingdom", "Canada", "Australia", 
            "Germany", "France", "Spain", "Italy", "Japan", "China",
            "India", "Brazil", "Mexico", "Russia", "South Africa"
        ]
        
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
        
        # For demo purposes, we'll simulate predictions
        # In production, you'd use your actual model
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'wav', 'mp3', 'm4a', 'flac'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Please upload WAV, MP3, M4A, or FLAC files.'}), 400
        
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
        
        # Create visualization
        visualization = create_audio_visualization(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'visualization': visualization,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/demo')
def demo():
    """Demo page with sample audio files"""
    return render_template('demo.html')

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