#!/bin/bash

# Flag Guessing Game Startup Script
echo "🎤 Flag Guessing Game - Audio Country Classifier"
echo "================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ to continue."
    exit 1
fi

# Check if models directory exists and has files
if [ ! -d "models" ]; then
    echo "❌ Models directory not found. Creating it..."
    mkdir -p models
fi

MODEL_FILE="models/audio_country_enhanced_transformer_model.keras"
ENCODER_FILE="models/label_encoder_enhanced_transformer.pkl"

if [ ! -f "$MODEL_FILE" ] || [ ! -f "$ENCODER_FILE" ]; then
    echo "⚠️  Model files are missing!"
    echo "Please copy your trained model files to the models/ directory:"
    echo "  - $MODEL_FILE"
    echo "  - $ENCODER_FILE"
    echo ""
    echo "See models/README.md for detailed instructions."
    echo ""
    echo "The application will start but prediction features won't work without the model files."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Model files found!"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Start the application
echo ""
echo "🚀 Starting the Flag Guessing Game..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python app.py