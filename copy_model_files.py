#!/usr/bin/env python3
"""
Script to help copy model files from Google Drive to local project
"""

import os
import shutil
import sys

def main():
    print("🤖 Model Files Setup Helper")
    print("=" * 40)
    
    # Expected model files (try multiple formats)
    model_files = {
        'model': ['audio_country_enhanced_transformer_model.keras', 'audio_country_enhanced_transformer_model.h5', 'model.keras', 'model.h5'],
        'encoder': ['label_encoder_enhanced_transformer.pkl', 'label_encoder.pkl', 'encoder.pkl'], 
        'scaler': ['feature_scaler_transformer.pkl', 'scaler.pkl']  # Optional
    }
    
    print("📁 Looking for model files in current directory...")
    
    files_found = []
    files_missing = []
    
    for file_type, filenames in model_files.items():
        found = False
        for filename in filenames:
            if os.path.exists(filename):
                size = os.path.getsize(filename) / (1024 * 1024)  # MB
                print(f"✅ {file_type.capitalize()}: {filename} ({size:.1f} MB)")
                files_found.append(filename)
                found = True
                break
        
        if not found:
            if file_type == 'scaler':
                print(f"⚠️  {file_type.capitalize()}: No scaler found (optional)")
            else:
                print(f"❌ {file_type.capitalize()}: No files found")
                files_missing.extend(filenames)
    
    print(f"\n📊 Summary:")
    print(f"   Found: {len(files_found)} files")
    print(f"   Missing: {len(files_missing)} files")
    
    if not files_missing:
        print("\n🎉 All model files are ready!")
        print("   You can now deploy to Railway!")
        return True
    
    print(f"\n🔧 To add your missing model files:")
    print("1. Download from Google Drive:")
    print("   - Go to your Google Drive")
    print("   - Navigate to: /content/drive/MyDrive/BrainInspiredMLFinalProject/")
    print("   - Download these files:")
    print("     • Your .keras model file (e.g., audio_country_enhanced_transformer_model.keras)")
    print("     • Your label encoder file (e.g., label_encoder_enhanced_transformer.pkl)")
    print("     • Feature scaler (optional - e.g., feature_scaler_transformer.pkl)")
    
    print("\n2. Copy to this directory:")
    print(f"   cd /Users/eishayadav/audio_classifier_demo")
    print("   cp /path/to/downloaded/your_model.keras .")
    print("   cp /path/to/downloaded/your_encoder.pkl .")
    print("   # Optional: cp /path/to/downloaded/your_scaler.pkl .")
    
    print("\n3. Or use environment variables:")
    print("   export MODEL_PATH=/path/to/your/model.keras")
    print("   export ENCODER_PATH=/path/to/your/encoder.pkl")
    print("   export SCALER_PATH=/path/to/your/scaler.pkl")
    
    print("\n4. After adding files, commit to GitHub:")
    print("   git add .")
    print("   git commit -m 'Add model files'")
    print("   git push origin main")
    
    return False

if __name__ == "__main__":
    main() 