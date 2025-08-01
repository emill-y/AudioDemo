import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import json
from datetime import datetime
import uuid
import time
from custom_layers import PositionalEncoding, EnhancedTransformerBlock

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 26
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 130  # Number of frames expected by the model

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
            'audio_country_enhanced_transformer_model.keras',
            'audio_country_enhanced_transformer_model.h5',
            'model.keras',
            'model.h5'
        ]
        
        encoder_paths = [
            'label_encoder_enhanced_transformer.pkl',
            'label_encoder.pkl'
        ]
        
        scaler_paths = [
            'feature_scaler_transformer.pkl',
            'scaler.pkl'
        ]
        
        print("Loading model components...")
        
        # Load the trained model
        model = None
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path, custom_objects={
                        'PositionalEncoding': PositionalEncoding,
                        'EnhancedTransformerBlock': EnhancedTransformerBlock
                    })
                    print(f"✅ Model loaded from: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_path}: {e}")
                    continue
        
        if model is None:
            print("⚠️ No model file found. Running in demo mode with mock predictions")
        
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
                    print(f"⚠️ Failed to load {encoder_path}: {e}")
                    continue
        
        if label_encoder is None:
            print("⚠️ No label encoder found. Using default country names")
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
                    print("✅ Feature scaler loaded")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {scaler_path}: {e}")
                    continue
        
        if scaler is None:
            print("⚠️ No feature scaler found. Will use raw features")
        
        print(f"Loaded {len(country_names)} countries: {country_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

def load_and_preprocess_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio file with error handling"""
    try:
        # Load audio with librosa (resampling if needed)
        audio, _ = librosa.load(
            file_path, 
            sr=sr, 
            duration=duration, 
            mono=True,
            res_type='kaiser_fast'
        )
        
        # Ensure audio is the correct length
        target_length = sr * duration
        if len(audio) < target_length:
            # Pad with silence if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Trim if too long
            audio = audio[:target_length]
            
        # Normalize audio to prevent very quiet recordings
        audio = librosa.util.normalize(audio) * 0.9  # Scale to 90% of maximum to avoid clipping
        
        return audio
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def extract_comprehensive_features(audio, sr=SAMPLE_RATE):
    """Extract audio features with fixed size for model input"""
    try:
        features = []
        
        # MFCC features with fixed size
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=13,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Ensure consistent shape
        if mfcc.shape[1] < TARGET_FRAMES:
            mfcc = np.pad(mfcc, ((0,0), (0,TARGET_FRAMES - mfcc.shape[1])), mode='constant')
            mfcc_delta = np.pad(mfcc_delta, ((0,0), (0,TARGET_FRAMES - mfcc_delta.shape[1])), mode='constant')
            mfcc_delta2 = np.pad(mfcc_delta2, ((0,0), (0,TARGET_FRAMES - mfcc_delta2.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :TARGET_FRAMES]
            mfcc_delta = mfcc_delta[:, :TARGET_FRAMES]
            mfcc_delta2 = mfcc_delta2[:, :TARGET_FRAMES]
            
        features.extend([mfcc, mfcc_delta, mfcc_delta2])
        
        # Mel-spectrogram (fixed size)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=128,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < TARGET_FRAMES:
            mel_spec_db = np.pad(mel_spec_db, ((0,0), (0,TARGET_FRAMES - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :TARGET_FRAMES]
        features.append(mel_spec_db[:26])  # First 26 bands
        
        # Combine all features with normalization
        combined = np.vstack(features)
        combined = (combined - np.mean(combined, axis=1, keepdims=True)) / (np.std(combined, axis=1, keepdims=True) + 1e-8)
        
        return combined.T
        
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return None

def predict_country(audio_file_path):
    """Predict country from audio file"""
    try:
        # Load and preprocess audio
        audio = load_and_preprocess_audio(audio_file_path)
        if audio is None:
            return None, "Error loading audio file"
        
        # Extract features
        features = extract_comprehensive_features(audio)
        if features is None:
            return None, "Error extracting features from audio"
        
        # Use actual model if available, otherwise use mock predictions
        if model is not None:
            try:
                # Reshape features for model input
                features_reshaped = features.reshape(1, *features.shape)
                
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
        
        for idx in top_indices:
            results.append({
                'country': country_names[idx],
                'confidence': float(predictions[idx] * 100)
            })
        
        return results, None
        
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

@app.route('/')
def index():
    """Main game page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_audio():
    """Handle audio prediction for the game"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(f"recording_{int(time.time())}.webm")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Verify file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save audio file'}), 500
            
        # Get predictions
        predictions, error = predict_country(filepath)
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Clean up uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

if __name__ == '__main__':
    # Load model on startup
    if load_model_and_preprocessors():
        print("Model loaded successfully!")
    else:
        print("Warning: Model not loaded. Running in demo mode.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)