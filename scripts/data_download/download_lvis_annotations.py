import os
import urllib.request

def download(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"✅ Already exists: {save_path}")
        return
    print(f"⬇️ Downloading {url} ...")
    urllib.request.urlretrieve(url, save_path)
    print(f"✅ Saved to {save_path}")

# LVIS official annotation URLs
lvis_annots = {
    "lvis_v1_train.json": "https://storage.googleapis.com/sfr-vision-language-research/LVIS/lvis_v1_train.json",
    "lvis_v1_val.json": "https://storage.googleapis.com/sfr-vision-language-research/LVIS/lvis_v1_val.json",
    "lvis_v1_image_info.json": "https://storage.googleapis.com/sfr-vision-language-research/LVIS/lvis_v1_image_info.json"
}

if __name__ == "__main__":
    base_dir = "data/lvis/annotations"
    for filename, url in lvis_annots.items():
        download(url, os.path.join(base_dir, filename))
