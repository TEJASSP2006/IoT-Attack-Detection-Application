"""
Train on All Dataset Parts
Combines all 170 CSV files and trains the model on the complete dataset
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


class CombinedDatasetTrainer:
    """Train on combined dataset parts"""
    
    def __init__(self):
        self.base_dir = r"E:\PE-2\archive\wataiData\csv\CICIoT2023"
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def find_all_parts(self):
        """Find all part files"""
        print("="*60)
        print("Finding All Dataset Parts")
        print("="*60)
        
        part_files = glob.glob(os.path.join(self.base_dir, "part-*.csv"))
        part_files.sort()
        
        if not part_files:
            print("❌ No part files found!")
            return None
        
        print(f"\n✓ Found {len(part_files)} part files")
        
        total_size = 0
        for file in part_files:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            total_size += size_mb
        
        print(f"✓ Total dataset size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        
        return part_files
    
    def load_and_combine_parts(self, part_files, sample_size=None):
        """
        Load and combine all part files
        
        Args:
            part_files: List of file paths
            sample_size: If set, sample this many rows from each file (for faster training)
        """
        print("\n" + "="*60)
        print("Loading and Combining Dataset Parts")
        print("="*60)
        
        if sample_size:
            print(f"\n⚠ Sample mode: Loading {sample_size} rows from each file")
        else:
            print("\n📊 Loading ALL data (this may take a while...)")
        
        all_data = []
        total_rows = 0
        
        for i, file in enumerate(part_files, 1):
            try:
                print(f"\n[{i}/{len(part_files)}] Loading: {os.path.basename(file)}")
                
                if sample_size:
                    # Load sample from each file
                    df = pd.read_csv(file, nrows=sample_size)
                else:
                    # Load entire file
                    df = pd.read_csv(file)
                
                print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
                total_rows += len(df)
                all_data.append(df)
                
            except Exception as e:
                print(f"  ❌ Error loading {file}: {e}")
                continue
        
        if not all_data:
            print("\n❌ No data loaded!")
            return None
        
        print("\n" + "="*60)
        print("Combining DataFrames...")
        print("="*60)
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✓ Combined dataset created!")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Total columns: {len(combined_df.columns)}")
        print(f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return combined_df
    
    def preprocess_data(self, df):
        """Preprocess the combined dataset"""
        print("\n" + "="*60)
        print("Preprocessing Data")
        print("="*60)
        
        # Find label column
        label_col = None
        for col in ['label', 'Label', 'attack_type', 'class']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            # Assume last column is label
            label_col = df.columns[-1]
            print(f"\n⚠ Using last column as label: {label_col}")
        else:
            print(f"\n✓ Label column found: {label_col}")
        
        # Show class distribution
        print(f"\nClass distribution:")
        class_counts = df[label_col].value_counts()
        for cls, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cls}: {count:,} ({percentage:.2f}%)")
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        print(f"\n✓ Features extracted: {len(X.columns)} numeric columns")
        self.feature_names = X.columns.tolist()
        
        # Handle missing and infinite values
        print("\nHandling missing and infinite values...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            print(f"  Found {missing_before:,} missing values")
            X = X.dropna()
            y = y[X.index]
            print(f"  ✓ Removed rows with missing values")
            print(f"  Remaining rows: {len(X):,}")
        else:
            print(f"  ✓ No missing values found")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n✓ Labels encoded")
        print(f"  Classes: {self.label_encoder.classes_.tolist()}")
        
        return X, y_encoded
    
    def train_model(self, X, y, test_size=0.2):
        """Train the model"""
        print("\n" + "="*60)
        print("Splitting and Scaling Data")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✓ Features scaled")
        
        # Train Random Forest
        print("\n" + "="*60)
        print("Training Random Forest Model")
        print("="*60)
        print("\nThis may take 15-60 minutes depending on dataset size...")
        print("Progress will be shown below:\n")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=2  # Show progress
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        print("\n✓ Training completed!")
        
        # Evaluate
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Feature importance
        print("\n" + "="*60)
        print("Top 20 Most Important Features")
        print("="*60)
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n" + feature_importance_df.head(20).to_string(index=False))
        
        return X_test_scaled, y_test
    
    def save_model(self, model_path='iot_attack_detector_full.pkl'):
        """Save the trained model"""
        print("\n" + "="*60)
        print("Saving Model")
        print("="*60)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        
        # Check file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"  File size: {size_mb:.2f} MB")
        
        # Rename to standard name so dashboard can use it
        if model_path != 'iot_attack_detector.pkl':
            import shutil
            shutil.copy(model_path, 'iot_attack_detector.pkl')
            print(f"✓ Also saved as: iot_attack_detector.pkl (for dashboard)")


def main():
    """Main training function"""
    print("="*60)
    print("CIC IoT 2023 - FULL DATASET TRAINING")
    print("Training on All 170 Parts")
    print("="*60)
    
    trainer = CombinedDatasetTrainer()
    
    # Find all parts
    part_files = trainer.find_all_parts()
    
    if not part_files:
        print("\n❌ No dataset parts found!")
        return
    
    # Ask user about sample mode
    print("\n" + "="*60)
    print("TRAINING OPTIONS")
    print("="*60)
    print("\n1. FULL DATASET (Recommended)")
    print("   - Uses all data from all 170 parts")
    print("   - Best accuracy")
    print("   - Takes 30-90 minutes")
    print("   - Requires ~4-8 GB RAM")
    print()
    print("2. SAMPLED DATASET (Faster)")
    print("   - Takes sample from each part")
    print("   - Faster training (10-20 minutes)")
    print("   - Good accuracy")
    print("   - Uses less memory")
    print()
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    sample_size = None
    if choice == "2":
        sample_str = input("How many rows per file? (recommend 5000-10000): ").strip()
        try:
            sample_size = int(sample_str)
            print(f"\n✓ Will sample {sample_size} rows from each file")
        except:
            print("\n⚠ Invalid input, using full dataset")
    
    # Load and combine
    combined_df = trainer.load_and_combine_parts(part_files, sample_size)
    
    if combined_df is None:
        print("\n❌ Failed to load data!")
        return
    
    # Preprocess
    X, y = trainer.preprocess_data(combined_df)
    
    # Free up memory
    del combined_df
    print("\n✓ Freed combined dataframe memory")
    
    # Train
    X_test, y_test = trainer.train_model(X, y)
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nYour model is ready to use!")
    print("\nNext steps:")
    print("  1. Launch dashboard: python web_dashboard.py")
    print("  2. Open browser: http://localhost:5000")
    print("  3. Start detecting attacks!")
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
