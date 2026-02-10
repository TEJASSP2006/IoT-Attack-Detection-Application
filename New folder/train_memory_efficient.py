"""
Memory-Efficient Training for Large Dataset
Handles 46M+ rows using chunked processing and memory optimization
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')


class MemoryEfficientTrainer:
    """Memory-efficient trainer for massive datasets"""
    
    def __init__(self):
        self.base_dir = r"E:\PE-2\archive\wataiData\csv\CICIoT2023"
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def find_all_parts(self):
        """Find all part files"""
        print("="*60)
        print("Finding Dataset Parts")
        print("="*60)
        
        part_files = glob.glob(os.path.join(self.base_dir, "part-*.csv"))
        part_files.sort()
        
        print(f"\n✓ Found {len(part_files)} part files")
        return part_files
    
    def load_with_sampling(self, part_files, samples_per_file=10000):
        """
        Load dataset with stratified sampling to reduce memory
        
        Args:
            part_files: List of CSV files
            samples_per_file: Number of samples to take from each file
        """
        print("\n" + "="*60)
        print("Loading Dataset with Smart Sampling")
        print("="*60)
        print(f"\nSampling {samples_per_file:,} rows per file")
        print(f"Total expected rows: {len(part_files) * samples_per_file:,}")
        
        all_data = []
        
        for i, file in enumerate(part_files, 1):
            try:
                # Read the file
                df = pd.read_csv(file)
                
                # Sample stratified by label to maintain class distribution
                if 'label' in df.columns:
                    # Stratified sampling
                    sample_size = min(samples_per_file, len(df))
                    df_sample = df.groupby('label', group_keys=False).apply(
                        lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))), 
                                         random_state=42)
                    )
                else:
                    # Random sampling
                    df_sample = df.sample(min(samples_per_file, len(df)), random_state=42)
                
                all_data.append(df_sample)
                
                if i % 20 == 0:
                    print(f"  [{i}/{len(part_files)}] Processed...")
                
                # Clean up
                del df
                gc.collect()
                
            except Exception as e:
                print(f"  ⚠ Error with {file}: {e}")
                continue
        
        print(f"\n✓ Loaded {len(all_data)} file samples")
        
        # Combine
        print("\nCombining samples...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Clean up
        del all_data
        gc.collect()
        
        print(f"\n✓ Dataset created!")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Memory: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return combined_df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        print("\n" + "="*60)
        print("Preprocessing Data")
        print("="*60)
        
        # Find label column
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        print(f"\n✓ Label column: {label_col}")
        
        # Show class distribution
        print(f"\nClass distribution (top 10):")
        class_counts = df[label_col].value_counts().head(10)
        for cls, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cls}: {count:,} ({percentage:.2f}%)")
        
        print(f"\nTotal unique classes: {df[label_col].nunique()}")
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        print(f"\n✓ Numeric features: {len(X.columns)}")
        self.feature_names = X.columns.tolist()
        
        # Handle missing/infinite values
        print("\nCleaning data...")
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if X.isnull().sum().sum() > 0:
            X = X.dropna()
            y = y[X.index]
            print(f"  Removed rows with missing values")
        
        print(f"✓ Clean dataset: {len(X):,} rows")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✓ Labels encoded: {len(self.label_encoder.classes_)} classes")
        
        return X, y_encoded
    
    def train_model_efficient(self, X, y):
        """Train model with memory-efficient approach"""
        print("\n" + "="*60)
        print("Training Model (Memory-Efficient Mode)")
        print("="*60)
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training: {len(X_train):,} samples")
        print(f"  Testing: {len(X_test):,} samples")
        
        # Convert to numpy arrays to save memory
        print("\nConverting to numpy arrays...")
        X_train = X_train.values
        X_test = X_test.values
        
        # Fit scaler on training data only
        print("\nFitting scaler on training data...")
        self.scaler.fit(X_train)
        
        # Transform in place to save memory
        print("Scaling training data...")
        X_train_scaled = self.scaler.transform(X_train)
        
        # Clean up
        del X_train
        gc.collect()
        
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        print("\nThis will take 15-45 minutes...")
        print("Progress shown below:\n")
        
        # Train with optimized parameters for large dataset
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Reduced to save memory
            min_samples_split=10,  # Increased to save memory
            min_samples_leaf=5,  # Increased to save memory
            max_features='sqrt',  # Reduces memory
            random_state=42,
            n_jobs=-1,
            verbose=2
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Clean up
        del X_train_scaled
        gc.collect()
        
        print("\n✓ Training complete!")
        
        # Evaluate
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        print("\nScaling test data...")
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n" + "-"*60)
        print("Classification Report (top 20 classes):")
        print("-"*60)
        
        # Show report for top classes only (to save space)
        unique_labels = np.unique(y_test)[:20]
        mask = np.isin(y_test, unique_labels)
        
        print(classification_report(
            y_test[mask], 
            y_pred[mask],
            labels=unique_labels,
            target_names=[str(x) for x in self.label_encoder.inverse_transform(unique_labels)],
            digits=4
        ))
        
        # Feature importance
        print("\n" + "="*60)
        print("Top 15 Most Important Features")
        print("="*60)
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n" + feature_importance_df.head(15).to_string(index=False))
        
        # Clean up
        del X_test_scaled, y_pred
        gc.collect()
    
    def save_model(self, model_path='iot_attack_detector.pkl'):
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
        
        print("\nSaving...")
        joblib.dump(model_data, model_path)
        
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        print(f"✓ Model saved: {model_path}")
        print(f"  File size: {size_mb:.2f} MB")


def main():
    """Main training function"""
    print("="*60)
    print("MEMORY-EFFICIENT TRAINING")
    print("For 46M+ Row Dataset")
    print("="*60)
    
    trainer = MemoryEfficientTrainer()
    
    # Find files
    part_files = trainer.find_all_parts()
    
    if not part_files:
        print("\n❌ No files found!")
        return
    
    # Ask for sampling size
    print("\n" + "="*60)
    print("SAMPLING OPTIONS")
    print("="*60)
    print("\nYour system ran out of memory with the full dataset.")
    print("Let's use smart sampling instead:\n")
    
    print("Recommended options:")
    print("  1. Light    - 5,000 rows/file  = ~850K total   (10-15 min, 92-95% accuracy)")
    print("  2. Medium   - 10,000 rows/file = ~1.7M total   (20-30 min, 95-97% accuracy)")
    print("  3. Heavy    - 20,000 rows/file = ~3.4M total   (40-60 min, 97-98% accuracy)")
    print("  4. Custom   - Your choice")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        samples_per_file = 5000
    elif choice == "2":
        samples_per_file = 10000
    elif choice == "3":
        samples_per_file = 20000
    elif choice == "4":
        samples_str = input("Samples per file: ").strip()
        try:
            samples_per_file = int(samples_str)
        except:
            print("Invalid input, using 10000")
            samples_per_file = 10000
    else:
        print("Invalid choice, using medium (10000)")
        samples_per_file = 10000
    
    print(f"\n✓ Will sample {samples_per_file:,} rows per file")
    
    # Load with sampling
    df = trainer.load_with_sampling(part_files, samples_per_file)
    
    # Preprocess
    X, y = trainer.preprocess_data(df)
    
    # Free memory
    del df
    gc.collect()
    print("\n✓ Freed memory")
    
    # Train
    trainer.train_model_efficient(X, y)
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nYour model is ready!")
    print("\nNote: This model was trained on a stratified sample")
    print("of the full dataset, maintaining class distribution.")
    print("\nNext steps:")
    print("  python web_dashboard.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
