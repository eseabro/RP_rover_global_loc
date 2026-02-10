import os
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2

# --- CONFIGURATION ---
SOURCE_DIR = "gazebo_imgs"
OUTPUT_DIR = "gazebo_tiles"
TILE_SIZE = 768   # Changed from 1000 to standard YOLO size
OVERLAP = 100     # Add overlap so rocks on the cut line aren't lost

def normalize_tile(data):
    """Converts 16-bit raw data to 8-bit visible image using CPU (Numpy)."""
    # 1. Convert to float
    img = data.astype(np.float32)
    
    # 2. Percentile Clipping (CPU)
    # Ignores the darkest 1% and brightest 1% of pixels (noise/shadows)
    p1, p99 = np.percentile(img, (1, 99))
    
    # 3. Clip and Stretch
    img = np.clip(img, p1, p99)
    if p99 > p1:
        img = ((img - p1) / (p99 - p1)) * 255.0
    
    # 4. Convert to 8-bit
    return img.astype(np.uint8)

def slice_image(jp2_path):
    basename = os.path.splitext(os.path.basename(jp2_path))[0]
    
    try:
        with rasterio.open(jp2_path) as src:
            width = src.width
            height = src.height
            
            print(f"Processing {basename} ({width}x{height})...")
            
            # Generate coordinates
            cols = list(range(0, width, TILE_SIZE - OVERLAP))
            rows = list(range(0, height, TILE_SIZE - OVERLAP))
            
            saved_count = 0
            
            for row in rows:
                for col in cols:
                    try:
                        # Define the window request
                        # Note: Rasterio automatically truncates if this goes out of bounds
                        window = Window(col, row, TILE_SIZE, TILE_SIZE)
                        
                        # Read the data
                        data = src.read(1, window=window)
                        
                        # Filter: Skip empty chunks
                        if np.mean(data) < 10: 
                            continue 
                        
                        # Normalize 16-bit to 8-bit
                        img_8bit = normalize_tile(data)
                        
                        # --- NEW PADDING LOGIC ---
                        # Check if we got a partial tile (e.g. 640x118)
                        h, w = img_8bit.shape
                        if h < TILE_SIZE or w < TILE_SIZE:
                            # Create a black square of the target size
                            padded_img = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                            # Paste the image data into the top-left corner
                            padded_img[:h, :w] = img_8bit
                            img_8bit = padded_img
                        # -------------------------

                        tile_name = f"{basename}_r{row}_c{col}.png"
                        save_path = os.path.join(OUTPUT_DIR, tile_name)
                        cv2.imwrite(save_path, img_8bit)
                        saved_count += 1
                        
                    except Exception as e:
                        continue
            
            print(f"   ✅ Created {saved_count} tiles from {basename}")

    except Exception as e:
        print(f"   ❌ Error on {basename}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith("2020.jpg")]
    
    print(f"Found {len(files)} source images. Slicing into {TILE_SIZE}x{TILE_SIZE} tiles...")
    
    for f in files:
        slice_image(os.path.join(SOURCE_DIR, f))

if __name__ == "__main__":
    main()