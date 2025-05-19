import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info  # Qwen2.5-VL repo에서 제공되는 util
from peft import PeftModel

# ========================
# 1. 경로 설정
# ========================
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # LoRA의 base
MODEL_DIR = "/root/project/step1-vgjson-grounding-ve-merged"  # LoRA + tokenizer 저장된 경로
IMG_PATH = "/root/project/data/visual_genome/VG_100K/2.jpg"
PROMPT = "Who is wearing the yellow shorts?"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 2. Processor (tokenizer 포함) 불러오기
# ========================
processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    use_fast=True
)

# ========================
# 3. Base 모델 로딩 후 임베딩 확장
# ========================
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 중요: 학습 당시 special token 추가한 만큼 embedding 사이즈 늘려줌
base_model.resize_token_embeddings(len(processor.tokenizer))

# ========================
# 4. LoRA 어댑터 로딩
# ========================
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# ========================
# 5. Inference 준비
# ========================
image = Image.open(IMG_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
]

text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
    padding=True
).to(device)

# ========================
# 6. Generate
# ========================
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,  # 중요
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,  # 👈 필수
        no_repeat_ngram_size=3,  # 👈 반복 억제
        eos_token_id=processor.tokenizer.eos_token_id,
    )

# input prompt 길이만큼 잘라서 output만 추출
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

# ========================
# 7. 출력
# ========================
print("\n🧾 Model Output:\n")
print(output_text[0])
