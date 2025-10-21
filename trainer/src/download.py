# download.py
import os
import requests
import zipfile

# --- Configuration ---
DATA_DIR = "data"
URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K dataset...")
        r = requests.get(URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        print(f"Dataset saved to {zip_path}")
    else:
        print("Dataset already downloaded.")

    # Extract the ZIP
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    print(f"Dataset extracted to {DATA_DIR}/ml-100k")

if __name__ == "__main__":
    download_movielens()
