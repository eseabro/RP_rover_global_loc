import os
from roboflow import Roboflow

# --- CONFIGURATION ---
API_KEY = "BIwT1pwa8uXOomFasXuU"
PROJECT_NAME = "gazebo-oh2pj"  # Your project ID (found in the URL: app.roboflow.com/workspace/PROJECT-NAME)
WORKSPACE_NAME = "rocksrock" # Your workspace name (from the URL)
IMAGE_FOLDER = "gazebo_tiles2" # The folder with your PNGs

def upload_images():
    print("Connecting to Roboflow...")
    
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    except Exception as e:
        print(f"❌ Connection failed. Check your API Key and Project names.\nError: {e}")
        return

    files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total = len(files)
    print(f"Found {total} images. Starting upload...")

    success_count = 0
    
    for i, filename in enumerate(files):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            # Upload the single image
            # 'num_retry_uploads' handles network hiccups automatically
            project.upload(file_path, num_retry_uploads=3)
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"   [{success_count}/{total}] Uploaded...")
                
        except Exception as e:
            print(f"   ⚠️ Failed to upload {filename}: {e}")

    print(f"\n✅ Finished! Uploaded {success_count} of {total} images.")

if __name__ == "__main__":
    upload_images()