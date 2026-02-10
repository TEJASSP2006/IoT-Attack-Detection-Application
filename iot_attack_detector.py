"""
IoT Attack Detection Application
Using UNB CIC IoT Dataset 2023

This application detects and classifies 7 types of IoT attacks:
1. DDoS (Distributed Denial of Service)
2. DoS (Denial of Service)
3. Recon (Reconnaissance)
4. Web-based attacks
5. Brute Force
6. Spoofing
7. Mirai botnet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


class IoTAttackDetector:
    """
    Main class for IoT attack detection and classification
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.attack_categories = [
            'DDoS', 'DoS', 'Recon', 'Web-based', 
            'Brute Force', 'Spoofing', 'Mirai', 'Benign'
        ]
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the CIC IoT 2023 dataset
        
        Args:
            file_path: Path to the dataset CSV file
            
        Returns:
            X_train, X_test, y_train, y_test: Split and preprocessed data
        """
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        # Handle missing values
        print("\nHandling missing values...")
        df = df.dropna()
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Separate features and labels
        # Typically the last column is the label
        if 'label' in df.columns:
            label_col = 'label'
        elif 'Label' in df.columns:
            label_col = 'Label'
        else:
            # Assume last column is the label
            label_col = df.columns[-1]
            
        print(f"\nUsing '{label_col}' as label column")
        
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        self.feature_names = X.columns.tolist()
        print(f"\nNumber of features: {len(self.feature_names)}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nAttack types found: {self.label_encoder.classes_}")
        print(f"Class distribution:\n{pd.Series(y).value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest classifier
        """
        print("\n" + "="*60)
        print("Training Random Forest Classifier...")
        print("="*60)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
    def train_gradient_boosting(self, X_train, y_train):
        """
        Train Gradient Boosting classifier
        """
        print("\n" + "="*60)
        print("Training Gradient Boosting Classifier...")
        print("="*60)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        print("\n" + "-"*60)
        print("Confusion Matrix:")
        print("-"*60)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, 
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        print(cm_df)
        
        return accuracy, y_pred
    
    def get_feature_importance(self, top_n=20):
        """
        Get and display feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n" + "="*60)
            print(f"Top {top_n} Most Important Features:")
            print("="*60)
            print(feature_importance_df.head(top_n).to_string(index=False))
            
            return feature_importance_df
        else:
            print("Feature importance not available for this model")
            return None
    
    def predict_attack(self, network_traffic_features):
        """
        Predict attack type from network traffic features
        
        Args:
            network_traffic_features: Array or DataFrame of features
            
        Returns:
            Predicted attack type
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Scale features
        features_scaled = self.scaler.transform([network_traffic_features])
        
        # Predict
        prediction = self.model.predict(features_scaled)
        prediction_proba = self.model.predict_proba(features_scaled)
        
        # Decode prediction
        attack_type = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(prediction_proba) * 100
        
        return {
            'attack_type': attack_type,
            'confidence': confidence,
            'probabilities': dict(zip(
                self.label_encoder.classes_,
                prediction_proba[0] * 100
            ))
        }
    
    def save_model(self, model_path='iot_detector_model.pkl'):
        """
        Save trained model and preprocessors
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path='iot_detector_model.pkl'):
        """
        Load trained model and preprocessors
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"\nModel loaded from {model_path}")


def main():
    """
    Main execution function
    """
    print("="*60)
    print("IoT Attack Detection System")
    print("UNB CIC IoT Dataset 2023")
    print("="*60)
    
    # Initialize detector
    detector = IoTAttackDetector()
    
    # Dataset path - Updated with your actual CSV file location
    dataset_path = r"E:\PE-2\archive\wataiData\csv\CICIoT2023\part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Download the CIC IoT 2023 dataset from:")
    print("   https://www.unb.ca/cic/datasets/iotdataset-2023.html")
    print("2. Update 'dataset_path' variable with your dataset location")
    print("3. Run this script to train the model")
    print("="*60)
    
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = detector.load_and_preprocess_data(dataset_path)
    
    # Train model (choose one)
    detector.train_random_forest(X_train, y_train)
    # detector.train_gradient_boosting(X_train, y_train)
    
    # Evaluate model
    accuracy, predictions = detector.evaluate_model(X_test, y_test)
    
    # Show feature importance
    feature_importance = detector.get_feature_importance(top_n=20)
    
    # Save model
    detector.save_model('iot_attack_detector.pkl')
    
    # Example prediction
    print("\n" + "="*60)
    print("Example Prediction:")
    print("="*60)
    # Use actual feature values from your test set
    sample_features = X_test[0]
    result = detector.predict_attack(sample_features)
    print(f"Predicted Attack: {result['attack_type']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("\nAll probabilities:")
    for attack, prob in result['probabilities'].items():
        print(f"  {attack}: {prob:.2f}%")
    


if __name__ == "__main__":
    main()
