import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load VG objects.json
with open("data/visual_genome/objects.json", "r") as f:
    vg_objects = json.load(f)

# Containers
image_bbox_count = defaultdict(int)
image_class_set = defaultdict(set)
class_freq = Counter()
image_ids = set()

# Parse data
for entry in vg_objects:
    image_id = entry["image_id"]
    image_ids.add(image_id)
    for obj in entry["objects"]:
        class_name = obj["names"][0].strip().lower()
        class_freq[class_name] += 1
        image_bbox_count[image_id] += 1
        image_class_set[image_id].add(class_name)

# Basic stats
num_images = len(image_ids)
num_classes = len(class_freq)
num_bboxes = sum(image_bbox_count.values())
avg_bboxes_per_image = num_bboxes / num_images
avg_classes_per_image = sum(len(s) for s in image_class_set.values()) / num_images

# Threshold-based counts
images_with_5plus_classes = sum(1 for s in image_class_set.values() if len(s) >= 5)
images_with_10plus_classes = sum(1 for s in image_class_set.values() if len(s) >= 10)
images_with_5plus_bboxes = sum(1 for c in image_bbox_count.values() if c >= 5)
images_with_10plus_bboxes = sum(1 for c in image_bbox_count.values() if c >= 10)

# Summary printout
print("📊 Visual Genome Object Stats Summary")
print(f"Total Images: {num_images}")
print(f"Total Unique Classes: {num_classes}")
print(f"Total Bounding Boxes: {num_bboxes}")
print(f"Avg BBoxes per Image: {avg_bboxes_per_image:.2f}")
print(f"Avg Classes per Image: {avg_classes_per_image:.2f}")
print(f"Images with ≥5 Classes: {images_with_5plus_classes}")
print(f"Images with ≥10 Classes: {images_with_10plus_classes}")
print(f"Images with ≥5 BBoxes: {images_with_5plus_bboxes}")
print(f"Images with ≥10 BBoxes: {images_with_10plus_bboxes}")

# Top 30 classes by frequency
top_classes = class_freq.most_common(30)
classes, freqs = zip(*top_classes)

plt.figure(figsize=(14, 6))
sns.barplot(x=list(classes), y=list(freqs))
plt.xticks(rotation=90)
plt.title("Top 30 Most Frequent Object Classes in Visual Genome")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
