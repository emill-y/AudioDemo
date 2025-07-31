#!/usr/bin/env python3
"""
Simple run script for the Audio Classifier Demo
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Set default configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("🎵 Audio Classifier Demo Starting...")
    print(f"📍 Server: http://localhost:{port}")
    print(f"🔧 Debug Mode: {debug}")
    print("🚀 Press Ctrl+C to stop the server")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1) 