import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# 설정
MERGED_DIR = "/root/project/step1-vgjson-grounding-ve-merged"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) tokenizer 로드 & special token 확인
tokenizer = AutoTokenizer.from_pretrained(
    MERGED_DIR,
    trust_remote_code=True,
    use_fast=False  # slow tokenizer 로 안정적으로 로드
)
print(">> vocab size:", len(tokenizer))
print(">> special tokens:", tokenizer.additional_special_tokens)

# 2) processor 로드
processor = AutoProcessor.from_pretrained(
    MERGED_DIR,
    tokenizer=tokenizer,
    trust_remote_code=True,
    use_fast=False
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# 3) 모델 로드
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.to(device).eval()

# 만약 PEFT wrapper 붙인 채 저장하신 게 아니라면:
# base = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", ...)
# model = PeftModel.from_pretrained(base, MERGED_DIR, torch_dtype=torch.bfloat16)

# 4) embedding vs vocab 점검
emb_shape = model.get_input_embeddings().weight.shape
print(f">> embedding matrix shape: {emb_shape} (should be [{len(tokenizer)}, D])")

# 5) 간단 테스트: ID→토큰
sample_ids = emb_shape[0] - 5, emb_shape[0] - 4, emb_shape[0] - 3, emb_shape[0] - 2, emb_shape[0] - 1
print(">> last 5 ids converted to tokens:", tokenizer.convert_ids_to_tokens(sample_ids))
