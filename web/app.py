"""Flask web application for Face Verification System"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
import cv2
import numpy as np
from datetime import datetime
import base64
import io
from PIL import Image

from system import FaceVerificationSystem
from utils import get_logger

logger = get_logger('web_app')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_face_verification_secret_key_change_in_production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Face Verification System
verification_system = FaceVerificationSystem()
verification_system.initialize()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/verify', methods=['POST'])
def verify_face():
    """API endpoint for face verification"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Process frame using verification system
            result = verification_system.process_frame(image)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Convert image to base64 for display
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare response
            response = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'image_base64': image_base64,
                'detections': [],
                'recognitions': [],
                'liveness_check': False
            }
            
            # Process results
            if 'detections' in result:
                for detection in result['detections']:
                    response['detections'].append({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    })
            
            if 'recognitions' in result:
                for recognition in result['recognitions']:
                    response['recognitions'].append({
                        'user_id': recognition.get('user_id', 'unknown'),
                        'confidence': recognition.get('confidence', 0.0)
                    })
            
            if 'liveness_result' in result:
                response['liveness_check'] = result['liveness_result'].get('is_live', False)
                response['liveness_confidence'] = result['liveness_result'].get('confidence', 0.0)
            
            return jsonify(response)
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """API endpoint for user registration"""
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'image' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        user_id = data['user_id']
        image_base64 = data['image']
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Register user using recognition system
        success = verification_system.register_user(user_id, image)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'User {user_id} registered successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Registration failed'}), 500
            
    except Exception as e:
        logger.error(f"User registration error: {e}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """API endpoint to get registered users"""
    try:
        users = verification_system.get_registered_users()
        return jsonify({
            'success': True,
            'users': users,
            'count': len(users),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get users error: {e}")
        return jsonify({'error': f'Failed to get users: {str(e)}'}), 500

@app.route('/api/logs')
def get_logs():
    """API endpoint to get system logs"""
    try:
        log_file = 'logs/system.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Return last 100 lines
                return jsonify({
                    'success': True,
                    'logs': lines[-100:],
                    'total_lines': len(lines)
                })
        return jsonify({'success': True, 'logs': [], 'total_lines': 0})
    except Exception as e:
        logger.error(f"Get logs error: {e}")
        return jsonify({'error': f'Failed to get logs: {str(e)}'}), 500

@app.route('/api/system/status')
def get_system_status():
    """API endpoint to get system status"""
    try:
        status = verification_system.get_system_status()
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get system status error: {e}")
        return jsonify({'error': f'Failed to get system status: {str(e)}'}), 500

@app.route('/admin')
def admin_panel():
    """Admin panel"""
    return render_template('admin.html')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File size exceeds limit (16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)