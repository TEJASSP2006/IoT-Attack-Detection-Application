"""
Demo Script - IoT Attack Detection
This script demonstrates the application without requiring the full dataset
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def create_synthetic_dataset(n_samples=1000):
    """
    Create synthetic dataset for demonstration
    This mimics the structure of CIC IoT 2023 dataset
    """
    print("Creating synthetic dataset for demonstration...")
    
    # Define attack types
    attack_types = ['DDoS', 'DoS', 'Recon', 'Web-based', 'Brute-Force', 'Spoofing', 'Mirai', 'Benign']
    
    # Create features (46 features typical for CIC IoT)
    feature_names = [
        'flow_duration', 'Header_Length', 'Protocol_Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
        'fin_count', 'urg_count', 'rst_count',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
        'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
        'Tot_sum', 'Min', 'Max', 'AVG', 'Std', 'Tot_size', 'IAT',
        'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
    ]
    
    # Generate random data with patterns for different attacks
    data = []
    labels = []
    
    for _ in range(n_samples):
        attack_type = np.random.choice(attack_types)
        
        # Create different patterns for different attack types
        if attack_type == 'DDoS':
            # High rate, many packets
            features = np.random.rand(46) * 100
            features[4] = np.random.rand() * 1000 + 500  # High rate
            features[5] = np.random.rand() * 1000 + 500  # High Srate
            features[19] = 1  # HTTP flag
            
        elif attack_type == 'DoS':
            # Similar to DDoS but lower rate
            features = np.random.rand(46) * 100
            features[4] = np.random.rand() * 500 + 200
            features[5] = np.random.rand() * 500 + 200
            
        elif attack_type == 'Recon':
            # Many different ports, varied protocols
            features = np.random.rand(46) * 50
            features[2] = np.random.rand() * 10  # Various protocols
            features[19:33] = np.random.rand(14)  # Various protocol flags
            
        elif attack_type == 'Web-based':
            # HTTP/HTTPS focused
            features = np.random.rand(46) * 100
            features[19] = 1  # HTTP
            features[20] = 1  # HTTPS
            features[1] = np.random.rand() * 200 + 100  # Header length
            
        elif attack_type == 'Brute-Force':
            # Many connection attempts
            features = np.random.rand(46) * 100
            features[15] = np.random.rand() * 100 + 50  # syn_count
            features[23] = 1  # SSH or Telnet
            
        elif attack_type == 'Spoofing':
            # Unusual source patterns
            features = np.random.rand(46) * 100
            features[29] = 1  # ARP
            features[31] = 1  # ICMP
            
        elif attack_type == 'Mirai':
            # Botnet patterns
            features = np.random.rand(46) * 150
            features[4] = np.random.rand() * 800 + 300
            features[23] = 1  # Telnet
            
        else:  # Benign
            # Normal patterns - lower values
            features = np.random.rand(46) * 30
            features[19] = np.random.choice([0, 1])  # Sometimes HTTP
            features[26] = 1  # TCP
            
        data.append(features)
        labels.append(attack_type)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['label'] = labels
    
    print(f"✓ Created synthetic dataset with {n_samples} samples")
    print(f"  Features: {len(feature_names)}")
    print(f"  Attack types: {len(attack_types)}")
    
    return df, feature_names


def train_demo_model():
    """Train a demo model with synthetic data"""
    print("\n" + "="*60)
    print("TRAINING DEMO MODEL")
    print("="*60)
    
    # Create synthetic dataset
    df, feature_names = create_synthetic_dataset(n_samples=5000)
    
    # Display class distribution
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Prepare data
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_encoded)
    
    print("✓ Model trained successfully!")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    joblib.dump(model_data, 'iot_attack_detector.pkl')
    print("✓ Model saved to: iot_attack_detector.pkl")
    
    # Test predictions
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    # Make some predictions
    test_samples = X.iloc[:5].values
    test_samples_scaled = scaler.transform(test_samples)
    predictions = model.predict(test_samples_scaled)
    probabilities = model.predict_proba(test_samples_scaled)
    
    print("\nSample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        attack_type = label_encoder.inverse_transform([pred])[0]
        confidence = np.max(prob) * 100
        print(f"\nSample {i+1}:")
        print(f"  Predicted: {attack_type}")
        print(f"  Confidence: {confidence:.2f}%")
    
    return model, scaler, label_encoder, feature_names


def demo_prediction_examples():
    """Show example predictions"""
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model_data = joblib.load('iot_attack_detector.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    print("✓ Model loaded")
    
    # Create example attacks
    examples = {
        'DDoS Attack': np.array([50, 80, 5, 100, 900, 850, 700, 1, 5, 2, 3, 8, 0, 0, 
                                 10, 15, 2, 0, 3, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 
                                 0, 0, 0, 500, 10, 200, 80, 30, 1000, 50, 100, 80, 
                                 90, 40, 50, 70]),
        
        'Normal Traffic': np.array([10, 20, 1, 30, 50, 40, 45, 0, 1, 0, 1, 2, 0, 0,
                                    2, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                    0, 0, 0, 100, 5, 50, 20, 10, 200, 10, 20, 15,
                                    18, 8, 10, 12]),
        
        'Brute Force': np.array([5, 30, 2, 20, 150, 140, 130, 0, 8, 1, 0, 5, 0, 0,
                                 5, 12, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                                 0, 0, 0, 150, 8, 80, 35, 15, 300, 25, 40, 30,
                                 35, 20, 25, 28])
    }
    
    print("\n" + "-"*60)
    for example_name, features in examples.items():
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        attack_type = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities) * 100
        
        print(f"\n{example_name}:")
        print(f"  Detected as: {attack_type}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Show top 3 probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        print("  Top 3 predictions:")
        for idx in top_3_idx:
            attack = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx] * 100
            print(f"    {attack}: {prob:.2f}%")


def main():
    """Main demo execution"""
    print("="*60)
    print("IoT Attack Detection - DEMO MODE")
    print("="*60)
    print("\nThis demo creates a synthetic dataset and trains a model")
    print("for demonstration purposes. For production use, please use")
    print("the actual CIC IoT 2023 dataset.")
    print("="*60)
    
    input("\nPress Enter to continue...")
    
    # Train model with synthetic data
    model, scaler, label_encoder, feature_names = train_demo_model()
    
    # Show demo predictions
    demo_prediction_examples()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nYou can now:")
    print("1. Run the web dashboard: python web_dashboard.py")
    print("2. Access at: http://localhost:5000")
    print("\nNote: This model was trained on synthetic data.")
    print("For real attack detection, train with actual CIC IoT 2023 dataset.")
    print("="*60)


if __name__ == "__main__":
    main()
