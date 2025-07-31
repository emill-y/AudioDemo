#!/usr/bin/env python3
"""
Setup script to copy model files from Google Drive to local project
"""

import os
import shutil
import sys

def setup_model_files():
    """Copy model files from Google Drive to local project"""
    
    print("🤖 Setting up model files for Audio Classifier Demo")
    print("=" * 50)
    
    # Expected model files
    model_files = {
        'model': 'audio_country_enhanced_transformer_model.h5',
        'encoder': 'label_encoder_enhanced_transformer.pkl', 
        'scaler': 'feature_scaler_transformer.pkl'
    }
    
    # Google Drive paths (from your original code)
    google_drive_path = "/content/drive/MyDrive/BrainInspiredMLFinalProject"
    
    print("📁 Looking for model files...")
    
    files_found = []
    files_missing = []
    
    for file_type, filename in model_files.items():
        # Check if file exists in current directory
        if os.path.exists(filename):
            print(f"✅ {file_type.capitalize()}: {filename} (already in project)")
            files_found.append(filename)
        else:
            print(f"❌ {file_type.capitalize()}: {filename} (missing)")
            files_missing.append(filename)
    
    if not files_missing:
        print("\n🎉 All model files are ready!")
        return True
    
    print(f"\n📋 Missing files: {len(files_missing)}")
    for file in files_missing:
        print(f"   - {file}")
    
    print("\n🔧 To add your model files:")
    print("1. Copy your model files to this directory:")
    print(f"   - {model_files['model']}")
    print(f"   - {model_files['encoder']}")
    print(f"   - {model_files['scaler']}")
    
    print("\n2. Or set environment variables:")
    print("   export MODEL_PATH=/path/to/your/model.h5")
    print("   export ENCODER_PATH=/path/to/your/encoder.pkl")
    print("   export SCALER_PATH=/path/to/your/scaler.pkl")
    
    print("\n3. Or update the file paths in app.py")
    
    return False

def check_dependencies():
    """Check if required dependencies are installed"""
    
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'tensorflow',
        'librosa', 
        'numpy',
        'scikit-learn',
        'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def main():
    """Main setup function"""
    
    print("🎵 Audio Classifier Demo Setup")
    print("=" * 40)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model files
    model_ok = setup_model_files()
    
    print("\n" + "=" * 40)
    
    if deps_ok and model_ok:
        print("🎉 Setup complete! You can now run:")
        print("   python run.py")
        print("   http://localhost:5000")
    elif deps_ok:
        print("⚠️  Dependencies OK, but model files missing.")
        print("   The app will run in demo mode.")
        print("   Run: python run.py")
    else:
        print("❌ Setup incomplete. Please install dependencies first.")
        print("   Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 