from ultralytics import YOLO
from roboflow import Roboflow
import torch
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

API_KEY = "BIwT1pwa8uXOomFasXuU"
PROJECT_NAME = "gazebo-seg"  # Your project ID (found in the URL: app.roboflow.com/workspace/PROJECT-NAME)
WORKSPACE_NAME = "rocksrock" # Your workspace name (from the URL)

# --- 1. DATA DOWNLOAD (Paste your Roboflow code here) ---
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
dataset = project.version(1).download("yolov12")
# --- 2. TRAIN THE MODEL ---
def main():

    torch.cuda.empty_cache()
    model = YOLO('yolo26m-seg.pt')
    

    # Train the model
    results = model.train(
        data=f"{dataset.location}/data.yaml",  # Points to the downloaded dataset
        epochs=200,      # 100 loops over the data is a standard starting point
        imgsz=992,      # Must match the tile size you created earlier
        batch=1,         # Lower this to 2 or 1 if you get "Out of Memory" errors
        amp=True,
        name='mars_above_maps',
        device=0         # Use 0 for GPU. Change to 'cpu' if you don't have an NVIDIA GPU.
    )

if __name__ == '__main__':
    main()
