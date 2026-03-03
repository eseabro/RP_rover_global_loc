import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
import torch

def process_satellite_tiled(image_path, model_path, output_csv, output_image=None, tile_size=1500):
    """
    Runs YOLO on a massive image using tiling to avoid GPU crashes.
    Stitches the visual results back together into a full-resolution map.
    """
    
    # 1. Setup GPU Memory Management
    torch.cuda.empty_cache()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading Model: {model_path}")
    model = YOLO(model_path)
    
    # Load massive image into SYSTEM MEMORY (CPU)
    # This is fine; 6000x6000x3 is only ~100MB of RAM.
    full_img = cv2.imread(image_path)
    if full_img is None:
        print(f"Error: Could not load image {image_path}")
        return

    H_global, W_global = full_img.shape[:2]
    
    # Create a canvas to draw detections on (Clone of original)
    if output_image:
        final_canvas = full_img.copy()

    # 2. Global Camera Parameters
    # HFOV = 1.0 rad, Width = 6000 -> fx = 5491.5
    fx = 5491.5
    fy = 5491.5
    cx = W_global / 2.0
    cy = H_global / 2.0
    current_depth = 50.0 

    csv_rows = []
    csv_rows.append(['ID', 'Global_U', 'Global_V', 'Map_X', 'Map_Y', 'Width_m', 'Length_m'])

    global_rock_count = 0

    # 3. Tiling Loop
    print(f"Processing {W_global}x{H_global} image in {tile_size}x{tile_size} tiles...")
    
    for y in range(0, H_global, tile_size):
        for x in range(0, W_global, tile_size):
            
            # Define Tile Bounds
            x_end = min(x + tile_size, W_global)
            y_end = min(y + tile_size, H_global)
            
            # Extract Tile from CPU memory
            tile = full_img[y:y_end, x:x_end]
            
            # Skip empty edge cases
            if tile.shape[0] < 10 or tile.shape[1] < 10:
                continue
                
            # Run Inference on GPU (Small batch)
            # verbose=False keeps the terminal clean
            results = model(tile, imgsz=tile_size, verbose=False)
            
            # --- A. Stitching Visuals ---
            if output_image:
                # .plot() generates the BGR image with masks/boxes for this tile
                annotated_tile = results[0].plot()
                
                # Paste this annotated tile back onto the main CPU canvas
                final_canvas[y:y_end, x:x_end] = annotated_tile
            
            # --- B. Processing Data ---
            if results[0].boxes is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                
                for box in boxes:
                    # Tile-Relative Coords
                    x1_t, y1_t, x2_t, y2_t = map(int, box[:4])
                    
                    # Convert to GLOBAL Coords
                    u_global = (x1_t + x2_t) / 2.0 + x
                    v_global = (y1_t + y2_t) / 2.0 + y
                    
                    px_width = x2_t - x1_t
                    px_height = y2_t - y1_t
                    
                    # 3D Math
                    real_width_m = (px_width * current_depth) / fx
                    real_length_m = (px_height * current_depth) / fy
                    
                    # Map position (X=East, Y=South)
                    map_x_offset = ((u_global - cx) * current_depth) / fx
                    map_y_offset = ((v_global - cy) * current_depth) / fy
                    
                    csv_rows.append([
                        global_rock_count, 
                        int(u_global), int(v_global),
                        f"{map_x_offset:.2f}", f"{map_y_offset:.2f}",
                        f"{real_width_m:.2f}", f"{real_length_m:.2f}"
                    ])
                    global_rock_count += 1
            
            # Optional: Print progress
            print(f"  Processed Tile [{x}:{x_end}, {y}:{y_end}]")
            
            # Cleanup GPU memory for next tile
            del results
            torch.cuda.empty_cache()

    # 4. Save Results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        
    if output_image:
        cv2.imwrite(output_image, final_canvas)
        print(f"Saved visual map to {output_image}")
        
    print(f"Done! Found {global_rock_count} rocks. Data saved to {output_csv}")

if __name__ == "__main__":
    # Replace these paths with your actual files
    img_file = "gazebo_imgs/MarsYard2021_above.jpg"
    model_file = "above_rocks.pt" # Your trained model
    out_csv = "above_rock_analysis.csv"
    out_img = "rock_visual_masks.jpg"  # <--- New image output path
    
    try:
        process_satellite_tiled(img_file, model_file, out_csv, output_image=out_img, tile_size=1500)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
