import json
from collections import defaultdict, Counter
from tqdm import tqdm
from PIL import Image
import os

# 설정값
MIN_CLASS_FREQ = 20
MAX_PER_CLASS = 5000
TOP_K = 50
MIN_CLASSES_PER_IMAGE = 3
MIN_BBOX_PER_IMAGE = 5
IMG_DIR_1 = "data/visual_genome/VG_100K"
IMG_DIR_2 = "data/visual_genome/VG_100K_2"

# 로드
with open("data/visual_genome/objects.json", "r") as f:
    vg_objects = json.load(f)

with open("data/visual_genome/attributes.json", "r") as f:
    vg_attributes = json.load(f)

# 속성 정보 매핑: image_id -> object_id -> attributes
attribute_map = defaultdict(lambda: defaultdict(list))
for entry in vg_attributes:
    image_id = entry["image_id"]
    for attr in entry["attributes"]:
        obj_id = attr["object_id"]
        attr_list = attr.get("attributes", [])
        attribute_map[image_id][obj_id] = attr_list

# 클래스 분석
class_freq = Counter()
image_class_map = defaultdict(set)
image_bbox_map = defaultdict(list)

for entry in vg_objects:
    image_id = entry["image_id"]
    for obj in entry["objects"]:
        name = obj["names"][0].strip().lower()
        object_id = obj["object_id"]
        class_freq[name] += 1
        image_class_map[image_id].add(name)
        image_bbox_map[image_id].append((name, object_id, obj["x"], obj["y"], obj["w"], obj["h"]))

# 필터링 기준
valid_classes = {cls for cls, freq in class_freq.items() if freq >= MIN_CLASS_FREQ}
head_classes = {cls for cls, _ in class_freq.most_common(TOP_K)}
class_quota = defaultdict(int)

# 최종 학습 데이터
training_data = []
import random

for image_id, objects in tqdm(image_bbox_map.items()):
    cls_set = image_class_map[image_id]

    if len(cls_set) < MIN_CLASSES_PER_IMAGE:
        continue
    if all(cls in head_classes for cls in cls_set):
        if random.random() < 0.1:  # 10% 정도는 예외로 포함
            pass
        else:
            continue
    if len(objects) < MIN_BBOX_PER_IMAGE:
        continue

    image_path = None
    for folder in [IMG_DIR_1, IMG_DIR_2]:
        candidate_path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(candidate_path):
            image_path = candidate_path
            break
    if image_path is None:
        continue

    try:
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
    except:
        continue

    obj_texts = []
    for cls, object_id, x, y, w, h in objects:
        if cls not in valid_classes:
            continue
        if class_quota[cls] >= MAX_PER_CLASS:
            continue

        x1 = round(x / orig_w, 2)
        y1 = round(y / orig_h, 2)
        x2 = round((x + w) / orig_w, 2)
        y2 = round((y + h) / orig_h, 2)

        class_quota[cls] += 1

        attributes = attribute_map[image_id].get(object_id, [])
        attr_prefix = " ".join(attributes[:3]) + " " if attributes else ""
        obj_texts.append(f"{attr_prefix}{cls} at ({x1}, {y1}, {x2}, {y2})")

    if len(obj_texts) >= MIN_BBOX_PER_IMAGE:
        input_prompt = "Describe this image."
        objects_block = f"[objects] {', '.join(obj_texts)} [/objects]"
        output_text = f"""let me explain all objects in the image.
{objects_block}
[reasoning]
.
[/reasoning]
based on objects, the answer is:
[answer] .
[/answer]"""

        training_data.append({
            "image_id": image_id,
            "input": input_prompt,
            "output": output_text
        })

with open("data/processed/vg_step1_training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2)

print(f"✅ Step 1 학습용 샘플 {len(training_data)}개 저장됨 (속성 포함, 좌표: 상대 정규화)")
