from ultralytics import YOLO
from roboflow import Roboflow

API_KEY = "BIwT1pwa8uXOomFasXuU"
PROJECT_NAME = "rock_sim"  # Your project ID (found in the URL: app.roboflow.com/workspace/PROJECT-NAME)
WORKSPACE_NAME = "rocksrock" # Your workspace name (from the URL)

# --- 1. DATA DOWNLOAD (Paste your Roboflow code here) ---
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
dataset = project.version(2).download("yolov12")

# --- 2. TRAIN THE MODEL ---
def main():
    # Load a model
    # 'yolov8n.pt' = Nano (Fastest, least accurate) - Good for testing
    # 'yolov8m.pt' = Medium (Slower, more accurate) - Good for final results
    model = YOLO('yolo26n.pt')
    

    # Train the model
    results = model.train(
        data=f"{dataset.location}/data.yaml",  # Points to the downloaded dataset
        epochs=200,      # 100 loops over the data is a standard starting point
        imgsz=2304,      # Must match the tile size you created earlier
        batch=1,         # Lower this to 2 or 1 if you get "Out of Memory" errors
        name='mars_rocks_val',
        device=0         # Use 0 for GPU. Change to 'cpu' if you don't have an NVIDIA GPU.
    )

if __name__ == '__main__':
    main()