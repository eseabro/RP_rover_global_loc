import requests
import os
import shutil
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- CONFIGURATION ---
BASE_URL = "https://hirise-pds.lpl.arizona.edu/PDS/RDR/"
SUB_FOLDERS = ["AEB", "ESP", "PSP", "TRA"]
FILES_PER_FOLDER = 2  # Keep low (files are huge)
SAVE_DIR = "source_jp2s"

def get_links(url):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, 'html.parser')
        return [urljoin(url, a['href']) for a in soup.find_all('a') 
                if a.get('href') and a['href'].endswith('/') and 'Parent' not in a.text]
    except: return []

def find_red_jp2(obs_url):
    try:
        r = requests.get(obs_url, timeout=20)
        soup = BeautifulSoup(r.text, 'html.parser')
        for a in soup.find_all('a'):
            href = a.get('href')
            if href and href.lower().endswith('.jp2') and "RED" in href.upper():
                return urljoin(obs_url, href)
    except: return None

def download_file(url, local_path):
    # Downloads 100MB+ files safely
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print(f"   ✅ Downloaded: {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("🚀 Starting Full-Resolution Downloads...")
    
    for folder in SUB_FOLDERS:
        print(f"\nScanning {folder}...")
        orbits = get_links(urljoin(BASE_URL, f"{folder}/"))
        count = 0
        
        for orb in orbits:
            if count >= FILES_PER_FOLDER: break
            obs_list = get_links(orb)
            
            for obs in obs_list:
                if count >= FILES_PER_FOLDER: break
                jp2 = find_red_jp2(obs)
                if jp2:
                    filename = f"{folder}_{count+1}.jp2"
                    save_path = os.path.join(SAVE_DIR, filename)
                    if not os.path.exists(save_path):
                        print(f"   Downloading source [{count+1}/{FILES_PER_FOLDER}]...")
                        if download_file(jp2, save_path):
                            count += 1
                    else:
                        print(f"   Skipping {filename} (Already exists)")
                        count += 1

if __name__ == "__main__":
    main()