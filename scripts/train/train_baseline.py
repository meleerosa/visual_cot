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
from peft import PeftModel, PeftConfig
from torch.utils.data import Dataset
import wandb

# ========================
# 1. 실험 이름과 초기화
# ========================
STEP2_ADAPTER_PATH = "/root/project/step2-visualcot-finetune"
PROJECT_NAME = "qwen2.5vl_step3_vg"
EXPERIMENT_NAME = "step3-vg-qa-rel"

wandb.init(
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    config={
        "learning_rate": 2e-6,
        "epochs": 3,
        "batch_size": 4 * 4,
    }
)

# ========================
# 2. 모델 로딩 (Step2 adapter 포함)
# ========================
peft_config = PeftConfig.from_pretrained(STEP2_ADAPTER_PATH)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, STEP2_ADAPTER_PATH, is_trainable=True)
model.enable_input_require_grads()

# ========================
# 3. Processor 로딩
# ========================
processor = AutoProcessor.from_pretrained(
    peft_config.base_model_name_or_path, trust_remote_code=True, use_fast=True
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# ========================
# 4. 데이터셋 로딩
# ========================
DATA_PATH = "data/step3_vg/step3_vg_qa_rel.jsonl"

class Step3Dataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        image = Image.open(os.path.join("datasets", image_path)).convert("RGB").resize((448, 448))

        prompt = sample["prompt"]

        question = ""
        if "[question]" in prompt and "[/question]" in prompt:
            question = prompt.split("[question]")[1].split("[/question]")[0].strip()

        answer_part = prompt.split("[/question]")[-1].strip()

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"[question] {question} [/question]"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer_part}
            ]}
        ]
        return {"image": image, "messages": messages}

train_dataset = Step3Dataset(DATA_PATH)

# ========================
# 5. Collator 정의
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
# 6. 학습 설정 및 Trainer
# ========================
training_args = TrainingArguments(
    output_dir=EXPERIMENT_NAME,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-6,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=1,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    report_to="wandb",
    remove_unused_columns=False,
    label_names=["labels"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer
)

trainer.train()

# ========================
# 7. 모델 저장
# ========================
trainer.save_model(EXPERIMENT_NAME)
processor.save_pretrained(EXPERIMENT_NAME)