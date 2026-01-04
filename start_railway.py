#!/usr/bin/env python
"""
Railway startup wrapper - runs the full main.py Flask app.
"""

import os
import sys
import traceback

# Get port from environment (Railway sets this)
PORT = int(os.environ.get("PORT", 8000))
print(f"Starting Crypto Pulse API on port {PORT}")
print(f"Python: {sys.version}")
sys.stdout.flush()

try:
    # Add project root to path
    sys.path.insert(0, ".")

    # Mock pathway before any imports
    from unittest.mock import MagicMock

    mock_pw = MagicMock()
    mock_pw.Schema = type("Schema", (), {})
    mock_pw.Duration = MagicMock()
    mock_pw.DateTimeUtc = MagicMock()
    sys.modules["pathway"] = mock_pw
    print("✓ Pathway mocked")
    sys.stdout.flush()

    # Set environment
    os.environ["PORT"] = str(PORT)
    os.environ["HOST"] = "0.0.0.0"

    # Import Flask app directly
    print("Importing main module...")
    sys.stdout.flush()

    from main import create_api_app

    print("✓ main module imported")
    sys.stdout.flush()

    app = create_api_app()
    print("✓ Flask app created")
    sys.stdout.flush()

    # Run the app
    print(f"Starting Flask server on 0.0.0.0:{PORT}...")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)
