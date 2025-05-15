import json
from collections import Counter
import re

# 파일 경로
file_path = "data/processed/vg_step1_training_data.json"

# 로딩
with open(file_path, "r") as f:
    data = json.load(f)

# 클래스 추출용 정규식
pattern = re.compile(r"([a-z0-9_ ]+) at \([0-9., ]+\)")

class_counter = Counter()

# 데이터 순회하며 클래스 수 카운트
for sample in data:
    output = sample["output"]
    objects_section = re.findall(r"\[objects\](.*?)\[/objects\]", output, re.DOTALL)
    if objects_section:
        object_text = objects_section[0]
        classes = pattern.findall(object_text)
        class_counter.update(cls.strip() for cls in classes)

# 결과 출력
print(f"총 클래스 수: {len(class_counter)}\n")
print("상위 30개 클래스:")
for cls, count in class_counter.most_common(1000):
    
    print(f"{cls:20} : {count}")