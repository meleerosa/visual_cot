import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# =============================
# 1. Load Base Model + LoRA Adapter
# =============================
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH = "/root/project/step1-vgjson-grounding-lora/checkpoint-4100"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)

# =============================
# 2. Prepare Input
# =============================
IMG_PATH = "/root/project/data/visual_genome/VG_100K/2.jpg" 
prompt = "Describe this image."

image = Image.open(IMG_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

# =============================
# 3. Run Inference
# =============================
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
    padding=True,
).to(device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

# =============================
# 4. Output
# =============================
print("\n🧾 Model Output:\n")
print(output_text[0])
