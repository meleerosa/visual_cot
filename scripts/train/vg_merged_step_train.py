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
from transformers import BitsAndBytesConfig 

# ========================
# 1. 실험 초기화
# ========================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = "qwen2.5vl_merged_step_grounding"
EXPERIMENT_NAME = "merged-step-vgjson-grounding-cot-vqa-lora"
STEP1_SAVE_DIR = "./step1_model"

wandb.init(
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    config={
        "learning_rate": 5e-6,
        "epochs": 3,
        "efficient_batch_size": 8,
    }
)

# ========================
# 2. 모델 로딩 (Flash Attention 2 + 4bit 로딩 + PEFT)
# ========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.enable_input_require_grads()

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
)
model = get_peft_model(model, lora_config)
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ========================
# 3. 데이터 로딩
# ========================
STEP1_DATA_PATH = "data/processed/vg_step1_training_data.json"
STEP2_DATA_PATH = "data/processed/vg_step2_training_data.json"
VG_FOLDER_1 = "data/visual_genome/VG_100K"
VG_FOLDER_2 = "data/visual_genome/VG_100K_2"

with open(STEP1_DATA_PATH, "r", encoding="utf-8") as f:
    step1_data = json.load(f)
with open(STEP2_DATA_PATH, "r", encoding="utf-8") as f:
    step2_data = json.load(f)

step1_len = len(step1_data)
data = step1_data + step2_data


def try_load_image(image_id):
    for folder in [VG_FOLDER_1, VG_FOLDER_2]:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return Image.open(path).convert("RGB").resize((448, 448))
    raise FileNotFoundError(f"❌ 이미지 {image_id}.jpg를 두 폴더에서 찾을 수 없습니다.")


class VGDatasetMerged(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = try_load_image(sample["image_id"])

        messages = [
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


train_dataset = VGDatasetMerged(data)

# ========================
# 4. Collate 함수
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
        truncation=True
    )

    labels = encoded["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels

    return encoded

# ========================
# 5. TrainingArguments
# ========================
training_args = TrainingArguments(
    output_dir=EXPERIMENT_NAME,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-6,
    warmup_steps=200,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=1,
    lr_scheduler_type="cosine",
    report_to="wandb",
    remove_unused_columns=False,
    label_names=["labels"],
    optim="paged_adamw_8bit"
)

# ========================
# 6. Trainer 실행 (중간 저장: step1 끝나면 저장)
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer
)

step1_end_step = (step1_len * training_args.num_train_epochs) // (
    training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
)
step1_saved = False

for step, _ in enumerate(trainer.get_train_dataloader(), start=1):
    if step == 1:
        trainer.train(resume_from_checkpoint=None)

    if step >= step1_end_step and not step1_saved:
        print(f"✅ Step 1 학습 완료. 모델 저장 중: {STEP1_SAVE_DIR}")
        trainer.save_model(STEP1_SAVE_DIR)
        processor.save_pretrained(STEP1_SAVE_DIR)
        step1_saved = True
        break

# step2까지 포함한 전체 학습 계속
trainer.train()
trainer.save_model(EXPERIMENT_NAME)
processor.save_pretrained(EXPERIMENT_NAME)
