import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 1. 설정
MERGED_DIR = "/root/project/step1-vgjson-grounding-ve-merged"   # merge_and_unload() 후 저장한 디렉토리
IMG_PATH   = "/root/project/data/visual_genome/VG_100K/2.jpg"
PROMPT     = "Describe this image."
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Processor (tokenizer + vision prep) 불러오기
processor = AutoProcessor.from_pretrained(
    MERGED_DIR,
    trust_remote_code=True,
    use_fast=True,      # special tokens 안정적 로드를 위해 slow tokenizer 사용
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# 3. 병합된 모델 그대로 로드
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.to(device).eval()

# 4. Inference 입력 준비
image = Image.open(IMG_PATH).convert("RGB").resize((448, 448))
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": PROMPT},
    ],
}]

# chat template 렌더링 & vision info 처리
text_prompt    = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
vision_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_prompt],
    images=vision_inputs,
    videos=video_inputs,
    return_tensors="pt",
    padding=True,
).to(device)

# 5. Generate
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,             # 샘플링 모드로 바꿔서 다양성↑
        temperature=0.9,            # 생성 온도 조절
        top_p=0.9,                  # nucleus sampling
        repetition_penalty=1.1,     # 반복 억제
        no_repeat_ngram_size=3,     # 3-gram 이상 반복 금지
        eos_token_id=processor.tokenizer.eos_token_id
    )

# 6. 디코딩 & 출력
trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.tokenizer.batch_decode(
    trimmed,
    skip_special_tokens=False
)[0]
for tok in ["[objects]", "[/objects]", "[answer]", "[/answer]"]:
    tid = processor.tokenizer.convert_tokens_to_ids(tok)
    emb = model.get_input_embeddings()(torch.tensor([tid]).to(model.device))
    print(f"{tok} | norm: {emb.norm().item():.4f}")

print("📝 Model Output:\n", output)
