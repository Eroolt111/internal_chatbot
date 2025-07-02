#!/usr/bin/env python3

"""
Web interface runner for the chatbot.
Run this from the project root directory.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

#FLask
from app.web_app import app

if __name__ == '__main__':
    print("Starting Chatbot Web Interface...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(
        host='127.0.0.1',  # !!! Only accessible from local computer !!!
        port=5000,
        debug=True  
    )
