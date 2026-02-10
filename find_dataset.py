"""
Dataset Finder and Loader
Helps find CSV files in your dataset directory
"""

import os
import glob

def find_csv_files(directory):
    """Find all CSV files in the directory"""
    print(f"Searching for CSV files in: {directory}")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        print("\nPlease check:")
        print("1. The path is correct")
        print("2. You're using forward slashes (/) or double backslashes (\\\\)")
        print("3. The drive letter is correct")
        return None
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        # Try subdirectories
        csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ No CSV files found in this directory")
        print("\nPlease check:")
        print("1. The dataset has been extracted")
        print("2. CSV files are in this directory or subdirectories")
        return None
    
    print(f"✓ Found {len(csv_files)} CSV file(s):")
    print()
    
    for i, file in enumerate(csv_files, 1):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"{i}. {os.path.basename(file)}")
        print(f"   Path: {file}")
        print(f"   Size: {size_mb:.2f} MB")
        print()
    
    return csv_files


def select_dataset(csv_files):
    """Let user select which CSV file to use"""
    if len(csv_files) == 1:
        print(f"Using the only CSV file found: {csv_files[0]}")
        return csv_files[0]
    
    print("\nMultiple CSV files found. Which one do you want to use?")
    print("(Or you can combine them later)")
    print()
    
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {os.path.basename(file)}")
    
    print(f"{len(csv_files) + 1}. Combine all files")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(csv_files) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(csv_files):
                return csv_files[choice_num - 1]
            elif choice_num == len(csv_files) + 1:
                return csv_files  # Return all files for combining
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def update_training_script(dataset_path):
    """Update the iot_attack_detector.py with the correct path"""
    print("\n" + "="*60)
    print("Updating iot_attack_detector.py...")
    print("="*60)
    
    try:
        with open('iot_attack_detector.py', 'r') as f:
            content = f.read()
        
        # Find and replace the dataset path
        if 'dataset_path = "E:/PE-2/archive"' in content:
            # Replace with actual file path
            content = content.replace(
                'dataset_path = "E:/PE-2/archive"',
                f'dataset_path = r"{dataset_path}"'
            )
        else:
            print("⚠ Warning: Could not find the expected dataset path line")
            print("Please manually update line with dataset_path")
            return False
        
        with open('iot_attack_detector.py', 'w') as f:
            f.write(content)
        
        print(f"✓ Updated dataset_path to: {dataset_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return False


def main():
    """Main function"""
    print("="*60)
    print("CIC IoT 2023 Dataset Finder")
    print("="*60)
    print()
    
    # Your dataset directory
    dataset_dir = r"E:\PE-2\archive"
    
    # Find CSV files
    csv_files = find_csv_files(dataset_dir)
    
    if not csv_files:
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("\n1. Check if the dataset is extracted:")
        print(f"   Look inside: {dataset_dir}")
        print()
        print("2. If you see a ZIP file, extract it first")
        print()
        print("3. Expected file names might be:")
        print("   - CICIoT2023.csv")
        print("   - dataset.csv")
        print("   - Or multiple CSV files for different attack types")
        print()
        print("4. Once extracted, run this script again")
        return
    
    # Select which file to use
    selected = select_dataset(csv_files)
    
    if isinstance(selected, list):
        # User wants to combine files
        print("\n⚠ Combining multiple files is advanced.")
        print("For now, let's use the first file.")
        print("You can modify the code later to combine them.")
        selected = csv_files[0]
    
    print("\n" + "="*60)
    print("SELECTED DATASET")
    print("="*60)
    print(f"File: {os.path.basename(selected)}")
    print(f"Path: {selected}")
    print(f"Size: {os.path.getsize(selected) / (1024 * 1024):.2f} MB")
    
    # Update the training script
    if update_training_script(selected):
        print("\n" + "="*60)
        print("✓ READY TO TRAIN!")
        print("="*60)
        print("\nNext step:")
        print("  python iot_attack_detector.py")
        print()
        print("This will:")
        print("  1. Load the dataset")
        print("  2. Train the model (5-30 minutes)")
        print("  3. Save the trained model")
    else:
        print("\n⚠ Please manually update the dataset path in iot_attack_detector.py")
        print(f"Set dataset_path to: r\"{selected}\"")


if __name__ == "__main__":
    main()
