import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import random
import tempfile
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Audio processing parameters (same as your model)
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 26
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Load model and encoders
model = None
label_encoder = None
scaler = None

def load_model_and_encoders():
    global model, label_encoder, scaler
    try:
        model = tf.keras.models.load_model('models/audio_country_enhanced_transformer_model.keras')
        
        with open('models/label_encoder_enhanced_transformer.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Create a dummy scaler if not available
        scaler = StandardScaler()
        print("Model and encoders loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

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
        print(f"Error loading audio {file_path}: {e}")
        return None

def extract_comprehensive_features(audio, sr=SAMPLE_RATE):
    """Extract comprehensive audio features (same as your model)"""
    features = []

    # MFCC features with higher coefficients
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend([mfcc, mfcc_delta, mfcc_delta2])

    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db[:26])  # Take first 26 mel bands

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

# Country data with flag emojis
COUNTRIES = {
    'afghanistan': '🇦🇫',
    'albania': '🇦🇱',
    'algeria': '🇩🇿',
    'andorra': '🇦🇩',
    'angola': '🇦🇴',
    'argentina': '🇦🇷',
    'armenia': '🇦🇲',
    'australia': '🇦🇺',
    'austria': '🇦🇹',
    'azerbaijan': '🇦🇿',
    'bahrain': '🇧🇭',
    'bangladesh': '🇧🇩',
    'belarus': '🇧🇾',
    'belgium': '🇧🇪',
    'brazil': '🇧🇷',
    'bulgaria': '🇧🇬',
    'cambodia': '🇰🇭',
    'canada': '🇨🇦',
    'chile': '🇨🇱',
    'china': '🇨🇳',
    'colombia': '🇨🇴',
    'croatia': '🇭🇷',
    'cuba': '🇨🇺',
    'cyprus': '🇨🇾',
    'denmark': '🇩🇰',
    'egypt': '🇪🇬',
    'estonia': '🇪🇪',
    'finland': '🇫🇮',
    'france': '🇫🇷',
    'georgia': '🇬🇪',
    'germany': '🇩🇪',
    'greece': '🇬🇷',
    'hungary': '🇭🇺',
    'iceland': '🇮🇸',
    'india': '🇮🇳',
    'indonesia': '🇮🇩',
    'iran': '🇮🇷',
    'iraq': '🇮🇶',
    'ireland': '🇮🇪',
    'israel': '🇮🇱',
    'italy': '🇮🇹',
    'japan': '🇯🇵',
    'jordan': '🇯🇴',
    'kazakhstan': '🇰🇿',
    'kenya': '🇰🇪',
    'korea': '🇰🇷',
    'laos': '🇱🇦',
    'latvia': '🇱🇻',
    'lebanon': '🇱🇧',
    'libya': '🇱🇾',
    'lithuania': '🇱🇹',
    'luxembourg': '🇱🇺',
    'malaysia': '🇲🇾',
    'mexico': '🇲🇽',
    'morocco': '🇲🇦',
    'nepal': '🇳🇵',
    'netherlands': '🇳🇱',
    'norway': '🇳🇴',
    'pakistan': '🇵🇰',
    'poland': '🇵🇱',
    'portugal': '🇵🇹',
    'romania': '🇷🇴',
    'russia': '🇷🇺',
    'saudi arabia': '🇸🇦',
    'singapore': '🇸🇬',
    'slovakia': '🇸🇰',
    'slovenia': '🇸🇮',
    'south africa': '🇿🇦',
    'spain': '🇪🇸',
    'sweden': '🇸🇪',
    'switzerland': '🇨🇭',
    'syria': '🇸🇾',
    'thailand': '🇹🇭',
    'turkey': '🇹🇷',
    'ukraine': '🇺🇦',
    'united kingdom': '🇬🇧',
    'united states': '🇺🇸',
    'uzbekistan': '🇺🇿',
    'vietnam': '🇻🇳'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_country')
def get_random_country():
    """Get a random country for the game"""
    if label_encoder is None:
        # If model not loaded, return from our country list
        country = random.choice(list(COUNTRIES.keys()))
        flag = COUNTRIES.get(country, '🏳️')
    else:
        # Get from trained countries
        countries = label_encoder.classes_
        country = random.choice(countries).lower()
        flag = COUNTRIES.get(country, '🏳️')
    
    return jsonify({
        'country': country,
        'flag': flag
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict country from uploaded audio"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            
            # Process audio
            audio = load_and_preprocess_audio(tmp_file.name)
            if audio is None:
                return jsonify({'error': 'Failed to process audio'}), 400
            
            # Extract features
            features = extract_comprehensive_features(audio)
            features = features.reshape(1, features.shape[0], features.shape[1])
            
            # Normalize features (basic normalization since we don't have the exact scaler)
            features_reshaped = features.reshape(-1, features.shape[-1])
            features_normalized = (features_reshaped - features_reshaped.mean()) / (features_reshaped.std() + 1e-8)
            features = features_normalized.reshape(features.shape)
            
            # Make prediction
            prediction = model.predict(features)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            predicted_country = label_encoder.classes_[predicted_class_idx].lower()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return jsonify({
                'predicted_country': predicted_country,
                'confidence': confidence,
                'all_predictions': {
                    label_encoder.classes_[i].lower(): float(prediction[0][i]) 
                    for i in range(len(label_encoder.classes_))
                }
            })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None
    })

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Try to load model
    model_loaded = load_model_and_encoders()
    if not model_loaded:
        print("Warning: Model not loaded. Some features may not work.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)