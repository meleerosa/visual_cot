import json
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import os

# 파일 경로들
IMG_DIRS = [
    "/root/project/data/visual_genome/VG_100K",
    "/root/project/data/visual_genome/VG_100K_2"
]

with open("/root/project/data/visual_genome/objects.json", "r") as f:
    objects_data = json.load(f)

with open("/root/project/data/visual_genome/relationships.json", "r") as f:
    relationships_data = json.load(f)

with open("/root/project/data/visual_genome/question_answers.json", "r") as f:
    qa_data = json.load(f)

with open("/root/project/data/visual_genome/qa_to_region_mapping.json", "r") as f:
    qa_region_map = json.load(f)

with open("/root/project/data/visual_genome/attributes.json", "r") as f:
    attributes_data = json.load(f)

# === Helper ===
def normalize_bbox(x, y, w, h, img_w, img_h):
    return [
        round(x / img_w, 2),
        round(y / img_h, 2),
        round(w / img_w, 2),
        round(h / img_h, 2)
    ]

def get_image_size(image_id):
    for folder in IMG_DIRS:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            try:
                with Image.open(path) as img:
                    return img.size
            except:
                return None
    return None

# === Indexing ===
image_objects = defaultdict(list)
object_attributes = defaultdict(dict)

for item in tqdm(objects_data, desc="Indexing objects"):
    image_id = item["image_id"]
    for obj in item["objects"]:
        names = obj.get("names", [])
        name = names[0].strip() if names else None
        object_id = obj.get("object_id")
        x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 0), obj.get("h", 0)
        if name:
            image_objects[image_id].append({
                "name": name,
                "bbox_raw": [x, y, w, h],
                "object_id": object_id
            })

for item in tqdm(attributes_data, desc="Indexing attributes"):
    image_id = item["image_id"]
    for attr in item["attributes"]:
        obj_id = attr.get("object_id")
        attrs = attr.get("attributes", [])
        if obj_id is not None:
            object_attributes[image_id][obj_id] = attrs

image_relationships = defaultdict(list)
for item in tqdm(relationships_data, desc="Indexing relationships"):
    image_id = item["image_id"]
    for rel in item["relationships"]:
        subj = rel.get("subject", {}).get("name", "").strip()
        pred = rel.get("predicate", "").strip()
        obj = rel.get("object", {}).get("name", "").strip()
        if subj and pred and obj:
            image_relationships[image_id].append(f"{subj} - {pred} - {obj}")

# === CoT 생성 ===
entries = []
for image_entry in tqdm(qa_data, desc="Processing QAs"):
    qa_container_image_id = image_entry["id"]
    if not image_entry.get("qas"):
        continue

    for qa in image_entry["qas"]:
        qa_id = qa["qa_id"]
        region_image_id = qa_region_map.get(str(qa_id))
        if not region_image_id:
            continue

        dims = get_image_size(region_image_id)
        if not dims:
            continue

        img_w, img_h = dims
        objects = image_objects.get(region_image_id, [])
        relationships = image_relationships.get(region_image_id, [])

        object_descriptions = []
        for obj in objects:
            name = obj["name"]
            x, y, w, h = obj["bbox_raw"]
            bbox = normalize_bbox(x, y, w, h, img_w, img_h)
            obj_id = obj.get("object_id")
            attrs = object_attributes[region_image_id].get(obj_id, [])
            label = f"{', '.join(attrs)} {name}".strip() if attrs else name
            label = label.replace("  ", " ").strip()
            if not label:
                continue
            object_descriptions.append(f"{label} at {bbox}")

        if not object_descriptions or not relationships:
            continue

        object_text = "[objects] " + "; ".join(object_descriptions) + " [/objects]"
        reasoning_lines = [f"Fact {i+1}: {rel}" for i, rel in enumerate(relationships)]
        reasoning_text = "[reasoning]\n" + "\n".join(reasoning_lines) + "\n[/reasoning]"

        answer = (
            "Let me describe all objects in the image.\n"
            f"{object_text}\n"
            "Now, let's look at the relationships between the objects.\n"
            f"{reasoning_text}\n"
            "Based on these facts, the answer is:\n"
            f"[answer] {qa['answer']} [/answer]"
        )

        entries.append({
            "image_id": region_image_id,
            "input": qa["question"],
            "output": answer,
        })

# 저장
output_path = "/root/project/data/processed/vg_step2_training_data.json"
with open(output_path, "w") as f:
    json.dump(entries, f, indent=2)

print(f"✅ 최종 저장 완료: {len(entries)}개 샘플 (bbox 정규화 적용됨)")
