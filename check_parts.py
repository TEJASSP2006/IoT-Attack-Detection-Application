"""
Check for Multiple CSV Parts
CIC IoT 2023 dataset often comes in multiple parts
"""

import os
import glob
import pandas as pd

def find_all_parts():
    """Find all part files in the dataset directory"""
    base_dir = r"E:\PE-2\archive\wataiData\csv\CICIoT2023"
    
    print("="*60)
    print("Searching for all dataset parts...")
    print("="*60)
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return None
    
    # Find all part-*.csv files
    part_files = glob.glob(os.path.join(base_dir, "part-*.csv"))
    part_files.sort()  # Sort to maintain order
    
    if not part_files:
        print("No part files found")
        return None
    
    print(f"\n✓ Found {len(part_files)} part file(s):\n")
    
    total_size = 0
    for i, file in enumerate(part_files, 1):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        total_size += size_mb
        print(f"{i}. {os.path.basename(file)}")
        print(f"   Size: {size_mb:.2f} MB")
    
    print(f"\n📊 Total size: {total_size:.2f} MB")
    
    return part_files


def check_file_structure(file_path):
    """Check the structure of a CSV file"""
    print("\n" + "="*60)
    print("Analyzing CSV structure...")
    print("="*60)
    
    try:
        # Read just the first few rows
        df = pd.read_csv(file_path, nrows=5)
        
        print(f"\n✓ File loaded successfully!")
        print(f"\nColumns ({len(df.columns)}):")
        print(df.columns.tolist())
        
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check for label column
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if label_cols:
            print(f"\n✓ Label column found: {label_cols[0]}")
        else:
            print(f"\n⚠ No 'label' column found. Last column: {df.columns[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def recommendation():
    """Provide recommendation on what to do"""
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    print("\n📌 For Training:")
    print()
    print("Option 1: Use Single Part (FASTEST)")
    print("  - Current file (67 MB) is enough to start")
    print("  - Training will be faster (5-10 minutes)")
    print("  - Good for testing and initial development")
    print("  - Run: python iot_attack_detector.py")
    print()
    
    print("Option 2: Combine All Parts (BEST ACCURACY)")
    print("  - Use all dataset parts for maximum accuracy")
    print("  - Training will take longer (15-30+ minutes)")
    print("  - Better representation of all attack types")
    print("  - Requires combining files first")
    print()
    
    print("💡 SUGGESTED APPROACH:")
    print("  1. Start with single part (current setup)")
    print("  2. Train and test the model")
    print("  3. If accuracy is good (>90%), you're done!")
    print("  4. If you want better accuracy, combine all parts later")


def main():
    """Main function"""
    print("="*60)
    print("CIC IoT 2023 Dataset - Parts Checker")
    print("="*60)
    
    # Find all parts
    part_files = find_all_parts()
    
    if not part_files:
        print("\n⚠ Could not find dataset parts")
        return
    
    # Check structure of first file
    if part_files:
        current_file = r"E:\PE-2\archive\wataiData\csv\CICIoT2023\part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
        check_file_structure(current_file)
    
    # Provide recommendation
    recommendation()
    
    print("\n" + "="*60)
    print("✅ CURRENT STATUS")
    print("="*60)
    print()
    print("Your iot_attack_detector.py is configured with:")
    print(f"  File: part-00000-...csv (67 MB)")
    print()
    print("✓ Ready to train!")
    print()
    print("Run: python iot_attack_detector.py")
    print("="*60)


if __name__ == "__main__":
    main()
