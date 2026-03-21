import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg    

# --- CONFIGURATION ---
INPUT_CSV = 'marsyard_cnes2_sat.csv'
OUTPUT_CSV = 'marsyard_cnes2_sat_aligned.csv'
IMAGE_PATH = 'marsyard_cnes.jpeg' # <-- UPDATE THIS with your image file
# 1. The Rotation (in degrees)
# Positive = Counter-Clockwise (Standard ROS ENU)
# 1. Image & Physical Scale (The New Math!)
IMG_WIDTH_PX = 733
IMG_HEIGHT_PX = 508
IMG_WIDTH_M = 83.0   # Physical meters across the width of the image
IMG_HEIGHT_M = 57.0  # Physical meters across the height of the image
# Negative = Clockwise
ROTATION_DEG = 90.0

# 2. The Inversion (Flip)
# Set to -1 if an axis is mirrored 
FLIP_X = -1 
FLIP_Y = -1

# 3. The Translation (Shift)
# How far is the true center of the Mars Yard from the Gazebo (0,0) origin?
X_OFFSET = 7  # meters
Y_OFFSET = -40  # meters

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
        rock_u.append(u)
        rock_v.append(v)
        
        # 1. Rotate
        rot_x = (mx * cos_t) - (my * sin_t)
        rot_y = (mx * sin_t) + (my * cos_t)
        
        # 2. Flip
        flip_x = rot_x * FLIP_X
        flip_y = rot_y * FLIP_Y
        
        # 3. Translate to Gazebo
        final_x = flip_x + X_OFFSET
        final_y = flip_y + Y_OFFSET
        
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
plt.scatter(rock_u, rock_v, c='cyan', marker='.', alpha=0.8, label='Mapped Rocks')

# Plot the Gazebo Origin and Axes overlay
plt.scatter(origin_u, origin_v, c='red', marker='*', s=300, label='Gazebo Origin (0,0)', zorder=5)

# Draw the Gazebo X-axis (Red) and Y-axis (Green)
plt.plot([origin_u, x_axis_u], [origin_v, x_axis_v], c='red', linewidth=3, label='Gazebo +X (10m Forward)')
plt.plot([origin_u, y_axis_u], [origin_v, y_axis_v], c='green', linewidth=3, label='Gazebo +Y (10m Left)')

plt.title(f"Gazebo Coordinate System Overlay\nRot: {ROTATION_DEG}°, Offset: ({X_OFFSET}, {Y_OFFSET})")
plt.xlabel("Image U (Pixels)")
plt.ylabel("Image V (Pixels)")
plt.legend()
plt.show()