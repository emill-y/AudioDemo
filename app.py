import random
import pickle
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import pycountry
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------------------
# Global configuration -----------------------------------------------------------------
# --------------------------------------------------------------------------------------

SAMPLE_RATE = 22050  # Must match training script
DURATION = 3         # Seconds – keep identical to training script
N_MFCC = 26          # Number of MFCCs extracted inside feature extractor
N_MELS = 128         # Mel-spectrogram bands – used inside feature extractor
N_FFT = 2048
HOP_LENGTH = 512

MODEL_FILE = Path("audio_country_enhanced_transformer_model.keras")
ENCODER_FILE = Path("label_encoder_enhanced_transformer.pkl")
SCALER_FILE = Path("feature_scaler_transformer.pkl")

# --------------------------------------------------------------------------------------
# Model / helper loading ---------------------------------------------------------------
# --------------------------------------------------------------------------------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file {MODEL_FILE} not found. Make sure you copied it into the project directory.")

if not ENCODER_FILE.exists():
    raise FileNotFoundError(
        f"Encoder file {ENCODER_FILE} not found. Make sure you copied it into the project directory.")

# Some users may omit the scaler. We fall back to identity transform in that case.
if SCALER_FILE.exists():
    with open(SCALER_FILE, "rb") as f:
        feature_scaler: StandardScaler = pickle.load(f)
else:
    class _IdentityScaler:
        def transform(self, x):
            return x
    feature_scaler = _IdentityScaler()
    print("[WARNING] feature_scaler_transformer.pkl not found – proceeding without feature scaling.")

model = load_model(MODEL_FILE, compile=False)
with open(ENCODER_FILE, "rb") as f:
    label_encoder = pickle.load(f)

COUNTRIES = list(label_encoder.classes_)

# --------------------------------------------------------------------------------------
# Utility functions --------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def country_to_flag(country_name: str) -> str:
    """Convert full country name to its flag emoji. Fallback to plain text if mapping fails."""
    try:
        # pycountry needs the official name – we attempt fuzzy search
        result = pycountry.countries.search_fuzzy(country_name)[0]
        iso_code = result.alpha_2.upper()
        # Convert ISO code to regional indicator symbols
        emoji = chr(ord(iso_code[0]) - 65 + 0x1F1E6) + chr(ord(iso_code[1]) - 65 + 0x1F1E6)
        return emoji
    except Exception:
        return country_name  # Fallback


def load_and_preprocess_audio(wav: np.ndarray | list, sr:SAMPLE_RATE = SAMPLE_RATE, duration:DURATION = DURATION):
    """Pad / trim waveform to fixed length and return mono signal."""
    if wav is None:
        return None
    wav = np.asarray(wav, dtype=np.float32)

    target_len = int(sr * duration)
    if wav.shape[0] < target_len:
        wav = np.pad(wav, (0, target_len - wav.shape[0]))
    else:
        wav = wav[:target_len]
    return wav


def extract_comprehensive_features(audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Replicates training-time feature extraction. Output shape (timesteps, feature_dim)."""
    features = []

    # MFCC & deltas
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend([mfcc, mfcc_delta, mfcc_delta2])

    # Mel spectrogram (take first 26 bands like training code)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db[:N_MFCC])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    features.append(chroma)

    # Spectral metrics
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    features.extend([spectral_centroids, spectral_rolloff, spectral_bandwidth, zero_crossing_rate])

    # Harmonic & percussive MFCC
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

    # Combine & normalize like the training script
    combined = np.vstack(features)
    combined = (combined - combined.mean(axis=1, keepdims=True)) / (
        combined.std(axis=1, keepdims=True) + 1e-8)
    return combined.T  # (timesteps, feature_dim)


def prepare_input(audio: np.ndarray):
    """Transform raw waveform into model-ready tensor."""
    wav = load_and_preprocess_audio(audio)
    feats = extract_comprehensive_features(wav)

    # Ensure fixed length for model
    seq_expected, feat_dim_expected = model.input_shape[1:3]
    seq_current = feats.shape[0]

    if seq_current < seq_expected:
        pad_width = seq_expected - seq_current
        feats = np.pad(feats, ((0, pad_width), (0, 0)))
    else:
        feats = feats[:seq_expected]

    # Feature scaling ------------------------------------------------------
    feats_scaled = feature_scaler.transform(feats)

    return np.expand_dims(feats_scaled, axis=0)  # Shape (1, seq_len, feat_dim)

# --------------------------------------------------------------------------------------
# Game state management ----------------------------------------------------------------
# --------------------------------------------------------------------------------------

current_country = random.choice(COUNTRIES)


def new_round():
    global current_country
    current_country = random.choice(COUNTRIES)
    return f"<div style='font-size:72px; text-align:center;'>{country_to_flag(current_country)}</div>", "Record your answer &raquo;"


def classify(audio):
    if audio is None:
        return "Please record your answer first."

    # gradio microphone returns (sample_rate, data) tuple when using streaming=False
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
        audio_np = np.array(data)
    else:
        # Might be file path – load with librosa
        audio_np, _ = librosa.load(audio, sr=SAMPLE_RATE, duration=DURATION)

    inp = prepare_input(audio_np)
    preds = model.predict(inp, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_country = label_encoder.inverse_transform([pred_idx])[0]

    is_correct = (pred_country.lower() == current_country.lower())
    result_text = (
        f"🤖 I think you said <b>{pred_country}</b>." +
        (" ✅ Correct!" if is_correct else f" ❌ Incorrect – I was looking for <b>{current_country}</b>.")
    )
    return result_text

# --------------------------------------------------------------------------------------
# Gradio Interface ---------------------------------------------------------------------
# --------------------------------------------------------------------------------------

with gr.Blocks(title="Country-by-Voice Game – Transformer Audio Classifier") as demo:
    gr.Markdown("""
    # 🌍 Country-by-Voice Game
    Speak the name of the country shown by its flag. Our Transformer model will try to recognise what you said!
    """)

    flag_display = gr.HTML(value="", elem_id="flag-display")
    status_output = gr.HTML(value="Press **New Flag** to start", elem_id="status-output")

    with gr.Row():
        record_btn = gr.Audio(source="microphone", type="numpy", label="Your Answer (max 3 s)")
        submit_btn = gr.Button("🧠 Guess!")

    new_flag_btn = gr.Button("🎌 New Flag")

    submit_btn.click(classify, inputs=record_btn, outputs=status_output)
    new_flag_btn.click(new_round, inputs=None, outputs=[flag_display, status_output])

    # When the app launches we immediately start a new round
    demo.load(fn=lambda: new_round(), inputs=None, outputs=[flag_display, status_output])

if __name__ == "__main__":
    demo.launch()