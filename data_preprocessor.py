"""
Data Preprocessing Utility for CIC IoT 2023 Dataset

This script helps prepare the dataset for training:
- Handles missing values
- Removes duplicates
- Balances classes (optional)
- Feature engineering
- Data validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import os


class DataPreprocessor:
    """
    Preprocessor for CIC IoT 2023 dataset
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_dataset(self, file_path):
        """Load dataset from CSV"""
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def basic_info(self, df):
        """Display basic information about the dataset"""
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        print(f"Shape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        print(df.columns.tolist())
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    def check_missing_values(self, df):
        """Check and display missing values"""
        print("\n" + "="*60)
        print("MISSING VALUES CHECK")
        print("="*60)
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(f"⚠ Found {len(missing_df)} columns with missing values:")
            print(missing_df.to_string(index=False))
        else:
            print("✓ No missing values found!")
            
        return missing_df
    
    def handle_missing_values(self, df, strategy='drop'):
        """
        Handle missing values
        
        Args:
            strategy: 'drop', 'mean', 'median', 'mode'
        """
        print(f"\nHandling missing values using strategy: {strategy}")
        
        if strategy == 'drop':
            df_clean = df.dropna()
            print(f"✓ Dropped rows with missing values")
            print(f"  Records: {len(df)} → {len(df_clean)}")
            
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_clean = df.copy()
            for col in numeric_cols:
                if df[col].isnull().any():
                    df_clean[col].fillna(df[col].mean(), inplace=True)
            print(f"✓ Filled numeric columns with mean values")
            
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_clean = df.copy()
            for col in numeric_cols:
                if df[col].isnull().any():
                    df_clean[col].fillna(df[col].median(), inplace=True)
            print(f"✓ Filled numeric columns with median values")
            
        return df_clean
    
    def handle_infinite_values(self, df):
        """Replace infinite values with NaN, then handle them"""
        print("\nHandling infinite values...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        
        for col in numeric_cols:
            inf_mask = np.isinf(df[col])
            inf_count += inf_mask.sum()
            
        print(f"  Found {inf_count} infinite values")
        
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        print(f"✓ Handled infinite values")
        print(f"  Records: {len(df)} → {len(df_clean)}")
        
        return df_clean
    
    def check_class_distribution(self, df, label_column='label'):
        """Check and display class distribution"""
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION")
        print("="*60)
        
        if label_column not in df.columns:
            # Try to find label column
            label_column = [col for col in df.columns if 'label' in col.lower()]
            if label_column:
                label_column = label_column[0]
            else:
                label_column = df.columns[-1]  # Assume last column
                
        print(f"Using column: '{label_column}'")
        
        class_counts = df[label_column].value_counts()
        class_pct = (class_counts / len(df)) * 100
        
        class_df = pd.DataFrame({
            'Class': class_counts.index,
            'Count': class_counts.values,
            'Percentage': class_pct.values
        })
        
        print("\n" + class_df.to_string(index=False))
        
        # Check for imbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 10:
            print(f"\n⚠ Warning: Significant class imbalance detected!")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
            print(f"  Consider using class balancing techniques")
        
        return class_df, label_column
    
    def balance_classes(self, df, label_column='label', method='undersample'):
        """
        Balance classes in the dataset
        
        Args:
            method: 'undersample', 'oversample'
        """
        print(f"\n" + "="*60)
        print(f"BALANCING CLASSES - Method: {method}")
        print("="*60)
        
        classes = df[label_column].unique()
        
        if method == 'undersample':
            # Find minority class size
            min_class_size = df[label_column].value_counts().min()
            print(f"Undersampling all classes to: {min_class_size} samples")
            
            df_balanced = pd.DataFrame()
            for cls in classes:
                df_class = df[df[label_column] == cls]
                df_class_resampled = resample(
                    df_class,
                    n_samples=min_class_size,
                    random_state=42,
                    replace=False
                )
                df_balanced = pd.concat([df_balanced, df_class_resampled])
                
        elif method == 'oversample':
            # Find majority class size
            max_class_size = df[label_column].value_counts().max()
            print(f"Oversampling all classes to: {max_class_size} samples")
            
            df_balanced = pd.DataFrame()
            for cls in classes:
                df_class = df[df[label_column] == cls]
                df_class_resampled = resample(
                    df_class,
                    n_samples=max_class_size,
                    random_state=42,
                    replace=True
                )
                df_balanced = pd.concat([df_balanced, df_class_resampled])
        
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Balancing complete")
        print(f"  Records: {len(df)} → {len(df_balanced)}")
        print(f"\nNew class distribution:")
        print(df_balanced[label_column].value_counts())
        
        return df_balanced
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("\nChecking for duplicates...")
        
        duplicates = df.duplicated().sum()
        print(f"  Found {duplicates} duplicate rows")
        
        if duplicates > 0:
            df_clean = df.drop_duplicates()
            print(f"✓ Removed duplicates")
            print(f"  Records: {len(df)} → {len(df_clean)}")
            return df_clean
        else:
            print("✓ No duplicates found")
            return df
    
    def validate_data(self, df):
        """Validate dataset for common issues"""
        print("\n" + "="*60)
        print("DATA VALIDATION")
        print("="*60)
        
        issues = []
        
        # Check for constant columns
        constant_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"⚠ {len(constant_cols)} constant columns found")
            print(f"⚠ Constant columns: {constant_cols}")
        
        # Check for high cardinality
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() == len(df):
                issues.append(f"⚠ Column '{col}' has unique values for every row")
        
        # Check for negative values where they shouldn't be
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].min() < 0 and 'delta' not in col.lower():
                issues.append(f"⚠ Column '{col}' has negative values")
        
        if not issues:
            print("✓ No data quality issues found!")
        else:
            print(f"Found {len(issues)} potential issues:")
            for issue in issues:
                print(f"  {issue}")
    
    def save_processed_data(self, df, output_path):
        """Save processed dataset"""
        print(f"\nSaving processed dataset to: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} records")
        print(f"  File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")


def main():
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("CIC IoT 2023 Dataset Preprocessor")
    print("="*60)
    
    # Configuration
    input_file = "path_to_your_dataset.csv"
    output_file = "processed_dataset.csv"
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Update 'input_file' with your dataset path")
    print("2. Run this script to preprocess the data")
    print("3. Use the output file for training")
    print("="*60)
    
    # Uncomment below when you have the dataset
    """
    preprocessor = DataPreprocessor()
    
    # Load dataset
    df = preprocessor.load_dataset(input_file)
    
    # Basic information
    preprocessor.basic_info(df)
    
    # Check missing values
    preprocessor.check_missing_values(df)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='drop')
    
    # Handle infinite values
    df = preprocessor.handle_infinite_values(df)
    
    # Remove duplicates
    df = preprocessor.remove_duplicates(df)
    
    # Check class distribution
    class_dist, label_col = preprocessor.check_class_distribution(df)
    
    # Balance classes (optional - uncomment if needed)
    # df = preprocessor.balance_classes(df, label_col, method='undersample')
    
    # Validate data
    preprocessor.validate_data(df)
    
    # Save processed data
    preprocessor.save_processed_data(df, output_file)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Use '{output_file}' for model training")
    """


if __name__ == "__main__":
    main()
