import os
import requests
import zipfile
import io
import shutil

def download_arc_data():
    url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
    print(f"Downloading ARC-AGI dataset from {url}...")
    
    r = requests.get(url)
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print("Extracting...")
        z.extractall("temp_arc_data")
        
        # Move files
        # Source structure: ARC-AGI-master/data/training -> data/training
        
        base_src = "temp_arc_data/ARC-AGI-master/data"
        
        for split in ['training', 'evaluation']:
            src_dir = os.path.join(base_src, split)
            dst_dir = os.path.join("data", split)
            
            if os.path.exists(src_dir):
                print(f"Populating {dst_dir}...")
                os.makedirs(dst_dir, exist_ok=True)
                for file in os.listdir(src_dir):
                    if file.endswith(".json"):
                        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        
        # Cleanup
        print("Cleaning up...")
        shutil.rmtree("temp_arc_data")
        print("Done! Data is ready.")
        
    else:
        print(f"Failed to download: {r.status_code}")

if __name__ == "__main__":
    download_arc_data()
