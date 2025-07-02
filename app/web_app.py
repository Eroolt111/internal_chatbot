from flask import Flask, render_template, request, jsonify
import logging
import os
from .pipeline import ChatbotPipeline # Make sure pipeline is correctly imported
from .config import config
import re
import threading # Import threading

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variable to hold the pipeline instance
pipeline_instance = None
pipeline_lock = threading.Lock() # Use a lock for thread safety during reload

def initialize_pipeline():
    """Function to initialize or reinitialize the chatbot pipeline."""
    global pipeline_instance
    with pipeline_lock:
        try:
            logger.info("Initializing chatbot pipeline...")
            pipeline_instance = ChatbotPipeline()
            logger.info("ChatbotPipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotPipeline: {e}")
            pipeline_instance = None
            return False

# Initialize pipeline on startup
initialize_pipeline()


@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the frontend"""
    global pipeline_instance # Ensure you refer to the global instance
    if not pipeline_instance:
        return jsonify({
            'error': 'Chatbot pipeline not initialized',
            'response': 'Sorry, the chatbot is currently unavailable.'
        }), 500
    
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please enter a question.'
            }), 400
        
        logger.info(f"Processing question: {question}")
        
        # Use the global pipeline instance to get the response
        result = pipeline_instance.query_pipeline.run(input=question)
        cleaned = re.sub(r'^assistant:\s*', '', str(result), flags=re.IGNORECASE).strip()
        logger.info(f"Generated response: {cleaned}")
        
        return jsonify({
            'response': cleaned,
            'status': 'success'
        })
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': error_msg,
            'response': 'Sorry, I encountered an error while processing your question.'
        }), 500

@app.route('/api/reload_pipeline', methods=['POST'])
def reload_pipeline():
    """Endpoint to trigger a reload of the chatbot pipeline."""
    logger.info("Received request to reload chatbot pipeline.")
    if initialize_pipeline():
        return jsonify({
            'status': 'success',
            'message': 'Chatbot pipeline reloaded successfully.'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to reload chatbot pipeline.'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    global pipeline_instance
    return jsonify({
        'status': 'healthy',
        'pipeline_ready': pipeline_instance is not None
    })

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )