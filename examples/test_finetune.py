#!/usr/bin/env python3
"""
Unsloth Finetuning Test for AMD Strix Halo (gfx1151)

This script tests:
1. Model loading
2. LoRA adapter setup
3. Basic training step

Run with: python scripts/test_finetune.py
"""

import os
import sys

# Enable experimental flash attention for AMD
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

def main():
    print("=" * 60)
    print("   Unsloth Finetuning Test - Strix Halo (gfx1151)")
    print("=" * 60)
    
    # Step 1: Import
    print("\n[1/6] Importing libraries...")
    try:
        import torch
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        print(f"✓ Imports successful")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return 1
    
    # Step 2: Load model
    print("\n[2/6] Loading model...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-0.5B",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        print(f"✓ Model loaded")
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return 1
    
    # Step 3: Setup LoRA
    print("\n[3/6] Setting up LoRA adapter...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"✓ LoRA adapter configured")
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    except Exception as e:
        print(f"✗ LoRA setup failed: {e}")
        return 1
    
    # Step 4: Create dummy dataset
    print("\n[4/6] Creating test dataset...")
    try:
        # Simple instruction-following format
        data = [
            {"text": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>"},
            {"text": "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\nHello! How can I help you today?<|im_end|>"},
            {"text": "<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\nThe sky is typically blue during the day.<|im_end|>"},
            {"text": "<|im_start|>user\nCount to 5<|im_end|>\n<|im_start|>assistant\n1, 2, 3, 4, 5<|im_end|>"},
        ] * 10  # Repeat for more samples
        
        dataset = Dataset.from_list(data)
        print(f"✓ Dataset created: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return 1
    
    # Step 5: Setup trainer
    print("\n[5/6] Setting up trainer...")
    try:
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,  # Just 10 steps for testing
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=1,
            optim="adamw_8bit",
            save_strategy="no",
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            args=training_args,
        )
        print(f"✓ Trainer configured")
    except Exception as e:
        print(f"✗ Trainer setup failed: {e}")
        return 1
    
    # Step 6: Run training
    print("\n[6/6] Running training (10 steps)...")
    try:
        trainer_stats = trainer.train()
        print(f"✓ Training completed!")
        print(f"  Training loss: {trainer_stats.training_loss:.4f}")
        print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
        print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 60)
    print("   TEST PASSED! 🎉")
    print("=" * 60)
    print(f"""
Your Strix Halo system is ready for finetuning!

Quick stats:
  - Model: Qwen2.5-0.5B (4-bit)
  - LoRA rank: 16
  - Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB
  - Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB

With {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB VRAM, you can finetune models like:
  - Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B
  - Llama-3.1-8B, Llama-3.1-70B
  - Mistral-7B, Mixtral-8x7B

Try a larger model:
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="unsloth/Qwen2.5-7B",
      max_seq_length=4096,
      load_in_4bit=True,
  )
""")
    
    # Cleanup
    import shutil
    if os.path.exists("./test_output"):
        shutil.rmtree("./test_output")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
