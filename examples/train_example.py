#!/usr/bin/env python3
"""
Example: Finetune Qwen2.5-7B on AMD Strix Halo

This script demonstrates full finetuning workflow:
1. Load a 7B model in 4-bit
2. Setup LoRA adapters
3. Load training data
4. Train with SFTTrainer
5. Save and export model

Adjust parameters based on your VRAM and use case.
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "unsloth/Qwen2.5-7B"  # Or "unsloth/Llama-3.1-8B", etc.
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0

# Training Configuration
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_STEPS = -1  # Set to positive number to limit steps

OUTPUT_DIR = "./outputs/qwen2.5-7b-finetuned"

# ============================================================================
# Load Model
# ============================================================================

print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=torch.bfloat16,
)

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# Setup LoRA
# ============================================================================

print("Setting up LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"LoRA configured. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# Load Dataset
# ============================================================================

print("Loading dataset...")

# Example: Using Alpaca dataset
# Replace with your own dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Format function for instruction tuning
def format_prompt(example):
    """Format as ChatML"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        user_content = f"{instruction}\n\nInput: {input_text}"
    else:
        user_content = instruction
    
    text = f"""<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    return {"text": text}

dataset = dataset.map(format_prompt)
print(f"Dataset loaded: {len(dataset)} samples")

# ============================================================================
# Training
# ============================================================================

print("Setting up trainer...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    warmup_ratio=0.03,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    report_to="none",  # Or "wandb" if you want logging
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# ============================================================================
# Train!
# ============================================================================

print("Starting training...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning rate: {LEARNING_RATE}")
print("")

trainer_stats = trainer.train()

print("\nTraining complete!")
print(f"  Loss: {trainer_stats.training_loss:.4f}")
print(f"  Runtime: {trainer_stats.metrics['train_runtime']/60:.1f} minutes")
print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ============================================================================
# Save Model
# ============================================================================

print(f"\nSaving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ============================================================================
# Optional: Export to GGUF for llama.cpp
# ============================================================================

# Uncomment to export to GGUF format
# print("\nExporting to GGUF...")
# model.save_pretrained_gguf(
#     f"{OUTPUT_DIR}/gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
# )

print("\n✅ Done!")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"To use the model:")
print(f"""
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("{OUTPUT_DIR}")
""")
