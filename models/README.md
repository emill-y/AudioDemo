# 🤖 AI Model Files

This directory should contain your trained model files for the audio country classifier.

## Required Files

You need to copy the following files from your training environment:

1. **`audio_country_enhanced_transformer_model.keras`** - The main transformer model
2. **`label_encoder_enhanced_transformer.pkl`** - The label encoder for country names

## How to Add Model Files

### From Google Colab
If you trained your model in Google Colab:

```python
# In your Colab notebook, download the files
from google.colab import files

# Download the model
files.download('/content/drive/MyDrive/BrainInspiredMLFinalProject/audio_country_enhanced_transformer_model.h5')

# Download the label encoder
files.download('/content/drive/MyDrive/BrainInspiredMLFinalProject/label_encoder_enhanced_transformer.pkl')
```

### From Local Training
If you trained locally:

```bash
# Copy from your training directory
cp /path/to/your/model/audio_country_enhanced_transformer_model.keras ./models/
cp /path/to/your/model/label_encoder_enhanced_transformer.pkl ./models/
```

## File Format Notes

- The model file should be in Keras format (`.keras` or `.h5`)
- The label encoder should be a pickled scikit-learn LabelEncoder
- Make sure the files have the exact names shown above

## Security Note

⚠️ **Important**: These model files can be large (50-200MB+). Do not commit them to Git!

The `.gitignore` file should exclude:
```
models/*.keras
models/*.h5
models/*.pkl
```

## Verification

Once you've added the files, you can verify they're working by running:

```bash
python -c "
import tensorflow as tf
import pickle
import os

# Check model
if os.path.exists('models/audio_country_enhanced_transformer_model.keras'):
    model = tf.keras.models.load_model('models/audio_country_enhanced_transformer_model.keras')
    print(f'✅ Model loaded successfully! Input shape: {model.input_shape}')
else:
    print('❌ Model file not found')

# Check label encoder
if os.path.exists('models/label_encoder_enhanced_transformer.pkl'):
    with open('models/label_encoder_enhanced_transformer.pkl', 'rb') as f:
        encoder = pickle.load(f)
    print(f'✅ Label encoder loaded! Classes: {len(encoder.classes_)}')
    print(f'Sample countries: {encoder.classes_[:5]}...')
else:
    print('❌ Label encoder file not found')
"
```

## Example Directory Structure

After adding your files, this directory should look like:

```
models/
├── README.md
├── audio_country_enhanced_transformer_model.keras  # Your model file
└── label_encoder_enhanced_transformer.pkl         # Your label encoder
```