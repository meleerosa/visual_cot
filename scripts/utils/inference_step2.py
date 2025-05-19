import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# ========== 경로 설정 ==========
base_model_path = "/root/project/step1-vgjson-grounding-ve-merged"  # base + vision encoder 병합된 거
lora_model_path = "/root/project/step2-vgjson-visual-cot-ve"  # step2 학습한 LoRA 결과
image_path = "/root/project/data/visual_genome/VG_100K_2/51.jpg"  # 테스트용 이미지 ID 경로
input_text = "What is happening in the scene?"  # 너의 질문

# ========== 모델 + LoRA 로딩 ==========
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# ========== Processor 로딩 ==========
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.tokenizer.pad_token = processor.tokenizer.eos_token  # 안전장치

if "[objects]" not in processor.tokenizer.get_vocab():
    raise ValueError("Tokenizer에 special token이 안 들어있음. 저장 문제일 수 있음.")
# ========== 이미지 로딩 ==========
def load_image(path):
    return Image.open(path).convert("RGB").resize((448, 448))

image = load_image(image_path)

# ========== 메시지 구성 ==========
messages = [
    {"role": "system", "content": "You are a visual assistant who provides detailed reasoning before answering."},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": input_text}]}
]

# ========== 입력 텍스트/이미지 전처리 ==========
rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=rendered,
    images=[image],
    return_tensors="pt",
    padding=True
).to(model.device)

# ========== 생성 ==========
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

# ========== 출력 디코딩 ==========
output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
print(processor.tokenizer.special_tokens_map)
print("🧠 Model Output:\n")
print(output_text)
