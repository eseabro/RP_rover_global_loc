import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_DIR = "mars_training_tiles"    # Folder with all your PNGs
OUTPUT_DIR = "mars_dataset_split"     # Where the 3 new folders will be created

# Split Ratios (Must add up to 1.0)
TRAIN_RATIO = 0.70  # 70% for Training
VAL_RATIO   = 0.20  # 20% for Validation
TEST_RATIO  = 0.10  # 10% for Testing

def split_dataset():
    # 1. Setup Directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # 2. Get all images
    # We filter for .png (or .jpg) to avoid moving random system files
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(files)
    
    if total_files == 0:
        print("❌ No images found in source directory.")
        return

    # 3. SHUFFLE! (The most important step)
    # This mixes tiles from AEB, ESP, and TRA together randomly
    random.shuffle(files)
    
    # 4. Calculate split points
    train_count = int(total_files * TRAIN_RATIO)
    val_count = int(total_files * VAL_RATIO)
    # Test gets the remainder to ensure no rounding errors lose a file
    
    # Slice the list
    train_files = files[:train_count]
    val_files   = files[train_count : train_count + val_count]
    test_files  = files[train_count + val_count:]
    
    print(f"Total Images: {total_files}")
    print(f"Training:   {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Testing:    {len(test_files)} images")
    print("-" * 30)

    # 5. Move the files
    def move_files(file_list, destination_name):
        dest_path = os.path.join(OUTPUT_DIR, destination_name)
        for f in file_list:
            src = os.path.join(SOURCE_DIR, f)
            dst = os.path.join(dest_path, f)
            shutil.copy2(src, dst) # copy2 preserves metadata; use shutil.move to cut/paste

    print("Copying Training files...")
    move_files(train_files, 'train')
    
    print("Copying Validation files...")
    move_files(val_files, 'val')
    
    print("Copying Testing files...")
    move_files(test_files, 'test')

    print("\n✅ Done! Check the 'mars_dataset_split' folder.")

if __name__ == "__main__":
    split_dataset()