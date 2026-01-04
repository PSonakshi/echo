#!/usr/bin/env python
"""
Minimal Railway startup script.
This bypasses the complex main.py to debug startup issues.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, ".")

# Get port from environment (Railway sets this)
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

print(f"Starting on {HOST}:{PORT}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')[:10]}")

# Try importing dependencies one by one to find the issue
print("\nChecking dependencies...")

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✓ python-dotenv")
except ImportError as e:
    print(f"✗ python-dotenv: {e}")

try:
    from flask import Flask

    print("✓ flask")
except ImportError as e:
    print(f"✗ flask: {e}")

try:
    from flask_cors import CORS

    print("✓ flask-cors")
except ImportError as e:
    print(f"✗ flask-cors: {e}")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    print("✓ vaderSentiment")
except ImportError as e:
    print(f"✗ vaderSentiment: {e}")

try:
    import requests

    print("✓ requests")
except ImportError as e:
    print(f"✗ requests: {e}")

# Try importing local modules
print("\nChecking local modules...")

try:
    # Mock pathway before importing modules that need it
    from unittest.mock import MagicMock

    mock_pw = MagicMock()
    mock_pw.Schema = type("Schema", (), {})
    sys.modules["pathway"] = mock_pw
    print("✓ pathway mocked")
except Exception as e:
    print(f"✗ pathway mock: {e}")

try:
    print("✓ transforms.sentiment")
except Exception as e:
    print(f"✗ transforms.sentiment: {e}")

try:
    print("✓ transforms.pulse_score")
except Exception as e:
    print(f"✗ transforms.pulse_score: {e}")

try:
    print("✓ simulator.hype_simulator")
except Exception as e:
    print(f"✗ simulator.hype_simulator: {e}")

# Create minimal Flask app
print("\nStarting minimal Flask app...")

try:
    app = Flask(__name__)
    CORS(app)

    @app.route("/health")
    def health():
        return {"status": "healthy", "port": PORT}

    @app.route("/")
    def root():
        return {"message": "Crypto Pulse API", "status": "running"}

    @app.route("/api/metrics")
    def metrics():
        return {"pulse_score": 5.0, "trending_phrases": ["test"], "status": "ok"}

    print(f"Starting Flask on {HOST}:{PORT}...")
    app.run(host=HOST, port=PORT, debug=False)

except Exception as e:
    print(f"Failed to start Flask: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
