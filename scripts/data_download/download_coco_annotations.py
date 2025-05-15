# scripts/data_download/download_coco_annotations.py
import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url, dest):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    annotations_dir = "./data/coco/annotations"
    zip_path = "./data/coco/annotations_trainval2017.zip"
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    os.makedirs(annotations_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading annotations...")
        download_file(url, zip_path)
    else:
        print("Annotation zip already downloaded.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("Extracting annotations...")
        zip_ref.extractall(path="./data/coco")

    print("Done. You can now cry tears of productivity.")

if __name__ == "__main__":
    main()
