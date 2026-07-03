"""
Vercel serverless function entry point for Flask app
This file is the entry point for Vercel's serverless functions
"""
import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Flask app
from app import app

# Export app for Vercel
# Vercel will automatically detect and use the Flask app
application = app

# For local testing
if __name__ == "__main__":
    app.run(debug=True, port=5000)
