# Strix Halo Finetuning Benchmarks

Comparing **Unsloth** vs standard PyTorch finetuning on AMD Strix Halo (gfx1151).

Reference: [kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning)

## Hardware

- **APU:** AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151)
- **VRAM:** 96GB allocated
- **Kernel:** 6.17.8-300.fc43
- **PyTorch:** Nightly ROCm 7.1

---

## Baseline (kyuz0's results - standard PyTorch)

| Model | Full FT | LoRA | 8-bit + LoRA | QLoRA |
|-------|---------|------|--------------|-------|
| Gemma-3 1B-IT | 19 GB / 2m52s | 15 GB / 2m | 13 GB / 8m | 13 GB / 9m |
| Gemma-3 4B-IT | 46 GB / 9m | 30 GB / 5m | 21 GB / 41m | 13 GB / 9m |
| Gemma-3 12B-IT | 115 GB / 25m | 67 GB / 13m | 43 GB / 2h38m | 26 GB / 23m |
| Gemma-3 27B-IT | OOM | OOM | 32 GB unstable | 19 GB runs |
| GPT-OSS-20B (MXFP4) | - | 32-38 GB / ~1h | - | - |

---

## Unsloth Results (TODO)

| Model | LoRA | QLoRA | vs Baseline |
|-------|------|-------|-------------|
| Gemma-3 1B-IT | TBD | TBD | TBD |
| Gemma-3 4B-IT | TBD | TBD | TBD |
| Gemma-3 12B-IT | TBD | TBD | TBD |
| Gemma-3 27B-IT | TBD | TBD | - |
| Qwen2.5-0.5B | TBD | TBD | - |
| GPT-OSS-20B (MXFP4) | TBD | TBD | - |

---

## Test Configuration

- **Dataset:** TBD
- **Steps:** TBD  
- **Batch size:** TBD
- **Sequence length:** TBD

---

## Running Benchmarks

```bash
source activate.sh
python benchmarks/benchmark_unsloth.py --model "unsloth/gemma-3-1b-it"
```
