import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg    

# --- CONFIGURATION ---
INPUT_CSV = 'marsyardCNES_sat.csv'
OUTPUT_CSV = 'marsyardCNES_sat_adjusted.csv'
IMAGE_PATH = 'marsyardCNES_masks.jpg' # <-- UPDATE THIS with your image file
# 1. The Rotation (in degrees)
# Positive = Counter-Clockwise (Standard ROS ENU)
# 1. Image & Physical Scale (The New Math!)
IMG_WIDTH_PX = 6000#733
IMG_HEIGHT_PX = 6000#508
IMG_WIDTH_M = 83.0   # Physical meters across the width of the image
IMG_HEIGHT_M = 57.0  # Physical meters across the height of the image
# Negative = Clockwise
ROTATION_DEG = 90.0

# 2. The Inversion (Flip)
# Set to -1 if an axis is mirrored 
FLIP_X = 1 
FLIP_Y = 1

# 3. The Translation (Shift)
# How far is the true center of the Mars Yard from the Gazebo (0,0) origin?
X_OFFSET = -8.712 #7  # meters
Y_OFFSET = 41.5#-40  # meters

# --- PRE-COMPUTE MATH ---
theta = math.radians(ROTATION_DEG)
cos_t = math.cos(theta)
sin_t = math.sin(theta)

# Pixels Per Meter
PPM_X = IMG_WIDTH_PX / IMG_WIDTH_M
PPM_Y = IMG_HEIGHT_PX / IMG_HEIGHT_M

def map_to_pixel(map_x, map_y):
    """Converts a physical Map coordinate (0,0 at center) to Image Pixels."""
    u = (IMG_WIDTH_PX / 2.0) + (map_x * PPM_X)
    v = (IMG_HEIGHT_PX / 2.0) + (map_y * PPM_Y) # V goes down from top-left
    return u, v

def gazebo_to_pixel(gx, gy):
    """Converts a Gazebo coordinate back to Map coordinates, then to Pixels."""
    # 1. Undo Translate
    fx = (gx - X_OFFSET) / FLIP_X
    fy = (gy - Y_OFFSET) / FLIP_Y
    
    # 2. Undo Rotate (Negative Theta)
    map_x = fx * math.cos(-theta) - fy * math.sin(-theta)
    map_y = fx * math.sin(-theta) + fy * math.cos(-theta)
    
    # 3. Apply Pixel Scale
    return map_to_pixel(map_x, map_y)

# Lists for plotting rocks
rock_u, rock_v = [], []

# --- PROCESSING ---
with open(INPUT_CSV, mode='r') as infile, open(OUTPUT_CSV, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    count = 0
    for row in reader:
        mx = float(row['Map_X'])
        my = float(row['Map_Y'])
        
        # Get pixels for the overlay plot
        u, v = map_to_pixel(mx, my)
        
        # 1. Rotate
        rot_x = (mx * cos_t) - (my * sin_t)
        rot_y = (mx * sin_t) + (my * cos_t)
        
        # 2. Flip
        flip_x = rot_x * FLIP_X
        flip_y = rot_y * FLIP_Y
        
        # 3. Translate to Gazebo
        final_x = flip_x + X_OFFSET
        final_y = flip_y + Y_OFFSET
        rock_u.append(u)
        rock_v.append(v)
        
        row['Map_X'] = round(final_x, 4)
        row['Map_Y'] = round(final_y, 4)
        writer.writerow(row)
        count += 1

print(f"✅ Processed {count} rocks!")

# Find the Gazebo Origin (0,0) and the Axes in Pixel Space
origin_u, origin_v = gazebo_to_pixel(0.0, 0.0)
x_axis_u, x_axis_v = gazebo_to_pixel(10.0, 0.0) # 10 meters Forward
y_axis_u, y_axis_v = gazebo_to_pixel(0.0, 10.0) # 10 meters Left

# --- PLOTTING ---
print("Generating image overlay plot...")
plt.figure(figsize=(12, 10))

try:
    img = mpimg.imread(IMAGE_PATH)
    plt.imshow(img)
except FileNotFoundError:
    print(f"⚠️ Could not find image at '{IMAGE_PATH}'. Plotting rocks on a blank canvas.")
    plt.gca().invert_yaxis()
    plt.xlim(0, IMG_WIDTH_PX)
    plt.ylim(IMG_HEIGHT_PX, 0)

# Plot the raw rocks in pixel space

# Plot the rotated/flipped rocks
plt.scatter(rock_u, rock_v, c='blue', marker='.', s=100, alpha=0.8, label='Adjusted Rocks (Gazebo Frame)')

# Plot the Gazebo Origin (0,0)
plt.scatter(0, 0, c='red', marker='*', s=300, label='Gazebo Origin (0,0)', zorder=5)

# Add Gazebo frame axes lines
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.axvline(0, color='black', linewidth=1.5, linestyle='--')

plt.title(f"Final Rock Output in Gazebo Frame\nRot: {ROTATION_DEG}°, Flip X/Y: ({FLIP_X}, {FLIP_Y}), Offset: ({X_OFFSET}, {Y_OFFSET})")
plt.xlabel("Gazebo +X (meters) [Forward]")
plt.ylabel("Gazebo +Y (meters) [Left]")
plt.axis('equal')  # CRITICAL: Keeps physical proportions 1:1
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.show()