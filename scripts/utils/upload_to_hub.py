from datasets import Dataset
import json

# 경로 설정
json_path = "/root/project/data/processed/vg_step2_training_data.json"
repo_id = "Jimyeong1532/vg_visual_cot"

# JSON 로드 및 Dataset 생성
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# 푸시 (비공개 업로드 가능, 라이센스 설정은 따로 해야 함)
dataset.push_to_hub(repo_id, config_name="vg_step2", private=True)
