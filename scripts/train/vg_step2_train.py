import os
import json
from PIL import Image
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import wandb

# ========================
# 1. 실험 초기화
# ========================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = "qwen2.5vl_step2_cot_vqa"
EXPERIMENT_NAME = "step2-vgjson-visual-cot"

wandb.init(
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    config={
        "learning_rate": 5e-6,
        "epochs": 3,
        "batch_size": 8,
    }
)

# ========================
# 2. 모델 로딩
# ========================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.enable_input_require_grads()
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
special_tokens = {
    "additional_special_tokens": [
        "[objects]", "[/objects]",
        "[reasoning]", "[/reasoning]",
        "[answer]", "[/answer]"
    ]
}

num_added = processor.tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(processor.tokenizer))

print(f"✅ Special tokens added: {num_added}")
# ========================
# 3. LoRA 설정
# ========================
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules = [
        # 텍스트 영역
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # Vision transformer blocks
        "visual.blocks.*.attn.qkv",
        "visual.blocks.*.attn.proj",
        "visual.blocks.*.mlp.gate_proj",
        "visual.blocks.*.mlp.up_proj",
        "visual.blocks.*.mlp.down_proj",

        # Merger
        "visual.merger.mlp.0",
        "visual.merger.mlp.2"
    ]
)
model = get_peft_model(model, lora_config)

print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ========================
# 4. 데이터 로딩
# ========================
DATA_PATH = "data/processed/vg_step2_training_data.json"
VG_FOLDER_1 = "data/visual_genome/VG_100K"
VG_FOLDER_2 = "data/visual_genome/VG_100K_2"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def try_load_image(image_id):
    for folder in [VG_FOLDER_1, VG_FOLDER_2]:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return Image.open(path).convert("RGB").resize((448, 448))
    raise FileNotFoundError(f"❌ 이미지 {image_id}.jpg를 두 폴더에서 찾을 수 없습니다.")

class VGStep1Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = try_load_image(sample["image_id"])

        messages = [
            {
                "role": "system",
                "content": "You are a visual assistant who provides detailed reasoning before answering."
                },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["input"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["output"]}
                ]
            }
        ]
        return {"image": image, "messages": messages}

train_dataset = VGStep1Dataset(data)

# ========================
# 5. Collate 함수
# ========================
def collate_fn(batch):
    messages = [sample["messages"] for sample in batch]
    images = [sample["image"] for sample in batch]

    rendered = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]

    encoded = processor(
        text=rendered,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    labels = encoded["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels

    return encoded

# ========================
# 6. TrainingArguments
# ========================
training_args = TrainingArguments(
    output_dir=EXPERIMENT_NAME,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    report_to="wandb",
    remove_unused_columns=False,
    label_names=["labels"]
)

# ========================
# 7. Trainer 실행
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer
)

trainer.train()
processor.save_pretrained(EXPERIMENT_NAME)
model.save_pretrained(EXPERIMENT_NAME)