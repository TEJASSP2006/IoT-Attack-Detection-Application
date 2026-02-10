"""
Quick Model Verification and Info
Check the trained model and display information
"""

import joblib
import os

def check_model():
    """Check if model exists and display info"""
    
    model_file = "iot_attack_detector.pkl"
    
    print("="*60)
    print("Model Verification")
    print("="*60)
    
    if not os.path.exists(model_file):
        print(f"\n❌ Model file not found: {model_file}")
        print("\nThe model was trained but encountered an error during saving.")
        print("Please run: python train_memory_efficient.py")
        return False
    
    print(f"\n✓ Model file found: {model_file}")
    
    # Check file size
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
    
    # Try to load it
    try:
        print("\nLoading model...")
        model_data = joblib.load(model_file)
        
        print("✓ Model loaded successfully!")
        print("\nModel Information:")
        print(f"  Model type: {type(model_data['model']).__name__}")
        print(f"  Features: {len(model_data['feature_names'])}")
        print(f"  Attack classes: {len(model_data['label_encoder'].classes_)}")
        
        print("\nAttack Types Detected:")
        classes = model_data['label_encoder'].classes_
        
        # Show first 20 classes
        for i, cls in enumerate(classes[:20], 1):
            print(f"  {i}. {cls}")
        
        if len(classes) > 20:
            print(f"  ... and {len(classes) - 20} more")
        
        print("\nTop 10 Most Important Features:")
        importances = model_data['model'].feature_importances_
        feature_names = model_data['feature_names']
        
        # Sort by importance
        sorted_idx = importances.argsort()[::-1][:10]
        
        for i, idx in enumerate(sorted_idx, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        print("\n" + "="*60)
        print("✅ MODEL IS READY!")
        print("="*60)
        print("\nYou can now:")
        print("  1. Launch the dashboard: python web_dashboard.py")
        print("  2. Open browser: http://localhost:5000")
        print("  3. Start detecting IoT attacks!")
        print("\n" + "="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nThe model file may be corrupted.")
        print("Please retrain: python train_memory_efficient.py")
        return False


if __name__ == "__main__":
    check_model()
