from ultralytics import YOLO
import os

# 1. Load your trained model
# Make sure this points to your ACTUAL best.pt file
model = YOLO('runs/detect/mars_rocks_val8/weights/best.pt') 

# 2. Define your folder of NEW images
source_folder = "labeled_gazebo" 

# 3. Run Inference
# save_txt=True -> Saves the coordinates in YOLO format
# conf=0.25 -> Accepts anything it's 25% sure is a rock (Lower this to 0.15 if you want to catch EVERYTHING)
results = model.predict(
    source=source_folder, 
    save_txt=True, 
    save_conf=False,  # Roboflow doesn't need confidence scores in the txt file
    conf=0.25
)

print("✅ Predictions complete! Check the 'runs/detect/predict/labels' folder.")