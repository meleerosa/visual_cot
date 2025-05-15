import os
import json
import pickle
from pathlib import Path

def preprocess_refcoco(
    refcoco_split="refcoco", split_by="unc",
    save_path="data/refcoco/refcoco_processed.json",
    annotation_dir="data/refcoco/annotations"
):
    print(">> Loading annotation files...")

    refs_pkl = Path(annotation_dir) / f"refs({split_by}).p"
    anns_path = Path(annotation_dir) / "instances.json"
    imgs_path = Path(annotation_dir) / "images.json"

    with open(refs_pkl, "rb") as f:
        refs = pickle.load(f)

    with open(anns_path, "r") as f:
        anns = {ann["id"]: ann for ann in json.load(f)["annotations"]}

    with open(imgs_path, "r") as f:
        imgs = {img["id"]: img for img in json.load(f)}

    results = []
    for ref in refs:
        ann = anns[ref["ann_id"]]
        img = imgs[ref["image_id"]]
        for sentence in ref["sentences"]:
            results.append({
                "image_id": ref["image_id"],
                "file_name": img["file_name"],
                "sentence": sentence,
                "bbox": ann["bbox"],  # [x, y, w, h]
                "category_id": ann["category_id"]
            })

    print(f">> Processed {len(results)} samples. Saving to {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ RefCOCO JSON preprocessing complete!")

if __name__ == "__main__":
    preprocess_refcoco()
