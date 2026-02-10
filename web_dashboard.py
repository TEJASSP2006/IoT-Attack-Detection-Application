"""
Web Dashboard for IoT Attack Detection
Flask-based web application with real-time monitoring
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

app = Flask(__name__)

# Global detector instance
detector = None
attack_log = []

def load_detector():
    """Load the trained model"""
    global detector
    from iot_attack_detector import IoTAttackDetector
    
    detector = IoTAttackDetector()
    try:
        detector.load_model('iot_attack_detector.pkl')
        return True
    except:
        return False


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for attack prediction"""
    global detector, attack_log
    
    if detector is None or detector.model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        data = request.json
        features = data.get('features', [])
        
        if len(features) != len(detector.feature_names):
            return jsonify({
                'error': f'Expected {len(detector.feature_names)} features, got {len(features)}'
            }), 400
        
        # Make prediction
        result = detector.predict_attack(features)
        
        # Log the detection
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'attack_type': result['attack_type'],
            'confidence': result['confidence'],
            'is_attack': result['attack_type'] != 'Benign'
        }
        attack_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(attack_log) > 1000:
            attack_log.pop(0)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """API endpoint for batch prediction from CSV"""
    global detector
    
    if detector is None or detector.model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Ensure we have the right features
        if len(df.columns) != len(detector.feature_names):
            return jsonify({
                'error': f'Expected {len(detector.feature_names)} features'
            }), 400
        
        # Make predictions
        predictions = []
        for idx, row in df.iterrows():
            features = row.values
            result = detector.predict_attack(features)
            predictions.append({
                'index': idx,
                'attack_type': result['attack_type'],
                'confidence': result['confidence']
            })
        
        # Create statistics
        attack_counts = {}
        for pred in predictions:
            attack_type = pred['attack_type']
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
        
        return jsonify({
            'total_predictions': len(predictions),
            'predictions': predictions,
            'statistics': attack_counts
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics')
def statistics():
    """Get attack detection statistics"""
    global attack_log
    
    if not attack_log:
        return jsonify({
            'total_detections': 0,
            'attack_count': 0,
            'benign_count': 0,
            'attack_types': {}
        })
    
    attack_count = sum(1 for entry in attack_log if entry['is_attack'])
    benign_count = len(attack_log) - attack_count
    
    attack_types = {}
    for entry in attack_log:
        attack_type = entry['attack_type']
        attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
    
    return jsonify({
        'total_detections': len(attack_log),
        'attack_count': attack_count,
        'benign_count': benign_count,
        'attack_types': attack_types,
        'recent_detections': attack_log[-10:][::-1]  # Last 10, reversed
    })


@app.route('/api/model_info')
def model_info():
    """Get model information"""
    global detector
    
    if detector is None or detector.model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    return jsonify({
        'model_type': type(detector.model).__name__,
        'features_count': len(detector.feature_names),
        'attack_categories': detector.label_encoder.classes_.tolist(),
        'feature_names': detector.feature_names[:20]  # First 20 features
    })


@app.route('/api/clear_log')
def clear_log():
    """Clear attack log"""
    global attack_log
    attack_log = []
    return jsonify({'message': 'Log cleared'})


if __name__ == '__main__':
    print("="*60)
    print("IoT Attack Detection Web Dashboard")
    print("="*60)
    
    # Try to load model
    if load_detector():
        print("\n✓ Model loaded successfully!")
        print(f"✓ Monitoring {len(detector.label_encoder.classes_)} attack categories")
    else:
        print("\n⚠ Warning: Model not found!")
        print("Please train the model first using iot_attack_detector.py")
        print("The dashboard will still run but predictions won't work.")
    
    print("\n" + "="*60)
    print("Starting web server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
