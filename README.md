# PyTorch & ML on AMD Strix Halo (gfx1151)

Get PyTorch, Unsloth, and ML frameworks working on AMD Ryzen AI Max+ 395 (Strix Halo).

> **TL;DR:** Use **kernel 6.17.8-300.fc43** + **PyTorch nightly ROCm 7.1** + `amdgpu.cwsr_enable=0`

## 🚀 Quick Start

```bash
git clone https://github.com/1ordo/strix-halo-unsloth.git
cd strix-halo-unsloth
./scripts/setup.sh      # Automated setup
./scripts/diagnose.sh   # Check if everything works
```

Or use the [pre-built container](#-container).

---

## ⚠️ The Key Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `Memory access fault by GPU node-1` | Missing kernel param | Add `amdgpu.cwsr_enable=0` |
| `invalid device function` | ROCm 6.x lacks gfx1151 | Use PyTorch nightly ROCm 7.1 |
| Segfault in `libhsa-runtime64.so.1` | **Kernel regression** | **Use kernel 6.17.8-300.fc43** |

**Critical discovery** ([cdbb_writes](https://medium.com/@cdbb_writes/from-bare-metal-to-ai-powerhouse-setting-up-amds-strix-halo-a58a1f3bc675)): Kernels 6.17.9+ cause `GCVM_L2_PROTECTION_FAULT`. Stick to 6.17.8.

---

## 💻 Tested On

- **Device:** GMKtec NucBox EVO-X2
- **APU:** AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151)
- **RAM/VRAM:** 128GB unified / 96GB allocated to GPU
- **OS:** Fedora 43, Kernel 6.17.8-300.fc43

---

## 🚀 Setup

### 1. BIOS: Set VRAM

1. Enter BIOS → `Advanced` → `AMD CBS` → `NBIO Common Options` → `GFX Configuration`
2. Set `UMA Frame Buffer Size` → **96GB**
3. Save and reboot

### 2. Install Kernel 6.17.8

```bash
# Install and lock the working kernel
sudo dnf install kernel-6.17.8-300.fc43 kernel-devel-6.17.8-300.fc43
sudo dnf install python3-dnf-plugins-core
sudo dnf versionlock add kernel-6.17.8-300.fc43.x86_64
sudo grubby --set-default /boot/vmlinuz-6.17.8-300.fc43.x86_64
```

### 3. Add Kernel Parameters

```bash
sudo grubby --update-kernel=ALL --args="amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432 amdgpu.cwsr_enable=0"
sudo reboot
```

| Parameter | Purpose |
|-----------|---------|
| `amdgpu.cwsr_enable=0` | **Critical** - Disables compute wave save/restore (gfx1151 fix) |
| `amdgpu.gttsize=131072` | 128GB GTT size for large VRAM |
| `ttm.pages_limit=33554432` | 128GB TTM page limit |
| `amd_iommu=off` | Prevents memory access faults |

### 4. Verify Setup

```bash
uname -r                           # Should be 6.17.8-300.fc43.x86_64
rocminfo | grep gfx                # Should show gfx1151
rocm-smi --showmeminfo vram        # Should show ~96GB
```

### 5. Install PyTorch ROCm 7.1

```bash
python3.12 -m venv ~/rocm7_unsloth
source ~/rocm7_unsloth/bin/activate
pip install --upgrade pip
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True AMD Radeon 8060S
```

### 6. Install Unsloth

```bash
pip install numpy transformers datasets accelerate peft trl huggingface_hub sentencepiece protobuf safetensors
pip install --no-deps unsloth unsloth-zoo
pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"
pip install cut_cross_entropy msgspec triton
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1 --no-deps

# Verify
python -c "from unsloth import FastLanguageModel; print('OK')"
```

---

## 🐳 Container

Skip manual setup - container has PyTorch + Unsloth pre-installed:

```bash
podman build -t strix-halo-unsloth -f Containerfile .
podman run -it --rm \
    --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video --group-add render \
    -v ~/models:/models \
    strix-halo-unsloth
```

**Note:** Still need kernel 6.17.8 + boot params on host.

---

## 📊 Model Recommendations (96GB VRAM)

| Model | VRAM (LoRA) |
|-------|-------------|
| Qwen2.5-7B | ~8GB |
| Qwen2.5-32B | ~24GB |
| **Qwen2.5-72B** | ~48GB |
| **Llama-3.1-70B** | ~45GB |

---

## 📚 Resources

- [Strix Halo Wiki](https://strixhalo.wiki/)
- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- [scottt/rocm-TheRock](https://github.com/scottt/rocm-TheRock/releases) - Pre-built PyTorch wheels
- [Unsloth AMD Docs](https://unsloth.ai/docs/get-started/install/amd)
- [Framework Community Thread](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)

---

## 🔑 Key Takeaways

1. **Kernel 6.17.8** - Newer kernels have GPU compute regression
2. **ROCm 7.1 nightly** - Only version with gfx1151 support
3. **cwsr_enable=0** - Required kernel param for gfx1151

---

**Credits:** [cdbb_writes](https://medium.com/@cdbb_writes/from-bare-metal-to-ai-powerhouse-setting-up-amds-strix-halo-a58a1f3bc675) (kernel discovery), [kyuz0](https://github.com/kyuz0), [scottt](https://github.com/scottt), [lhl](https://strixhalo.wiki/)

---

**Credits:** [cdbb_writes](https://medium.com/@cdbb_writes/from-bare-metal-to-ai-powerhouse-setting-up-amds-strix-halo-a58a1f3bc675) (kernel discovery), [kyuz0](https://github.com/kyuz0), [scottt](https://github.com/scottt), [lhl](https://strixhalo.wiki/)
