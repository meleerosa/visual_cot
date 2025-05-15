import json
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

# Load LVIS annotation file
lvis_path = "data/lvis/annotations/lvis_v1_train.json"
with open(lvis_path, "r") as f:
    lvis_data = json.load(f)

# Extract components
annotations = lvis_data["annotations"]
images = lvis_data["images"]
categories = lvis_data["categories"]

# Initialize
cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
category_counts = defaultdict(int)
image_box_counts = defaultdict(int)
image_to_classes = defaultdict(set)
bbox_area_list = []

# Count and aggregate
for ann in tqdm(annotations, desc="Processing Annotations"):
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    x, y, w, h = ann["bbox"]

    category_counts[category_id] += 1
    image_box_counts[image_id] += 1
    image_to_classes[image_id].add(category_id)
    bbox_area_list.append(w * h)

# Analysis
total_annotations = len(annotations)
total_images = len(images)
total_categories = len(categories)
avg_bboxes_per_image = sum(image_box_counts.values()) / total_images
annotation_file_size_mb = os.path.getsize(lvis_path) / (1024 * 1024)
bbox_area_mean = np.mean(bbox_area_list)
bbox_area_std = np.std(bbox_area_list)


# 이미지 중 바운딩 박스 class가 5개 이상인 이미지 수
num_images_with_many_classes = sum(1 for class_set in image_to_classes.values() if len(class_set) >= 5)

# 요약
summary = {
    "Total Categories": total_categories,
    "Total Annotations": total_annotations,
    "Total Images": total_images,
    "Mean BBoxes per Image": round(avg_bboxes_per_image, 2),
    "Annotation File Size (MB)": round(annotation_file_size_mb, 2),
    "BBox Area Mean": round(bbox_area_mean, 2),
    "BBox Area Std": round(bbox_area_std, 2),
    "Images with ≥5 unique classes": num_images_with_many_classes
}

# 출력
print("\n📊 LVIS 분석 요약:")
for k, v in summary.items():
    print(f"{k}: {v}")
