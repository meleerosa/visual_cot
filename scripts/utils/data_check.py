import os
import json
from tqdm import tqdm

# 경로 설정
DATA_PATH = "data/processed/vg_step1_training_data.json"
IMG_FOLDER_1 = "data/visual_genome/VG_100K"
IMG_FOLDER_2 = "data/visual_genome/VG_100K_2"
OUTPUT_PATH = "data/processed/vg_step1_training_data.filtered.json"

# 실제 존재하는 이미지 ID 수집
def get_available_image_ids():
    img_files_1 = {f.split('.')[0] for f in os.listdir(IMG_FOLDER_1) if f.endswith(".jpg")}
    img_files_2 = {f.split('.')[0] for f in os.listdir(IMG_FOLDER_2) if f.endswith(".jpg")}
    return img_files_1.union(img_files_2)

available_image_ids = get_available_image_ids()
print(f"✅ 총 사용 가능한 이미지 수: {len(available_image_ids)}")

# 전처리된 학습 데이터 로딩
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📦 전처리된 샘플 수: {len(data)}")

# 존재하지 않는 이미지 ID 샘플 필터링
missing = []
valid_samples = []

for sample in tqdm(data, desc="🔍 이미지 존재 여부 확인 중"):
    img_id_str = str(sample["image_id"])
    if img_id_str in available_image_ids:
        valid_samples.append(sample)
    else:
        missing.append(img_id_str)

# 결과 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(valid_samples, f, indent=2)

# 요약 출력
print("\n🎯 필터링 요약")
print(f"❌ 존재하지 않는 이미지 ID 수: {len(missing)}")
print(f"✅ 최종 학습 가능 샘플 수: {len(valid_samples)}")
print(f"📄 저장 경로: {OUTPUT_PATH}")
