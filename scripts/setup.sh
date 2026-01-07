#!/bin/bash
# Strix Halo Setup Script
# Installs kernel 6.17.8, configures boot params, sets up PyTorch + Unsloth
#
# Usage: ./scripts/setup.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

VENV_PATH="$HOME/strix-halo-unsloth"
KERNEL="6.17.8-300.fc43.x86_64"

echo "═══════════════════════════════════════════════════════════"
echo "  Strix Halo AI Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check not root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Don't run as root. Script will sudo when needed.${NC}"
    exit 1
fi

# Check Fedora
if ! grep -q "Fedora" /etc/os-release 2>/dev/null; then
    echo -e "${YELLOW}Warning: Tested on Fedora 43 only.${NC}"
fi

# Step 1: Kernel
echo -e "${YELLOW}[1/4] Kernel setup${NC}"
if [[ "$(uname -r)" == "$KERNEL" ]]; then
    echo -e "${GREEN}✓ Already on $KERNEL${NC}"
else
    echo "Installing kernel $KERNEL..."
    sudo dnf install -y kernel-$KERNEL kernel-devel-$KERNEL
    sudo dnf install -y python3-dnf-plugins-core
    sudo dnf versionlock add kernel-$KERNEL 2>/dev/null || true
    sudo grubby --set-default /boot/vmlinuz-$KERNEL
    echo -e "${GREEN}✓ Kernel installed. Reboot required after setup.${NC}"
fi

# Step 2: Boot params
echo ""
echo -e "${YELLOW}[2/4] Boot parameters${NC}"
if grep -q "amdgpu.cwsr_enable=0" /proc/cmdline 2>/dev/null; then
    echo -e "${GREEN}✓ Already configured${NC}"
else
    echo "Adding boot parameters..."
    sudo grubby --update-kernel=ALL --args="amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432 amdgpu.cwsr_enable=0"
    echo -e "${GREEN}✓ Boot params added. Reboot required.${NC}"
fi

# Step 3: Python venv + PyTorch
echo ""
echo -e "${YELLOW}[3/4] PyTorch ROCm 7.1${NC}"
if [[ -d "$VENV_PATH" ]]; then
    echo "Using existing venv: $VENV_PATH"
else
    echo "Creating venv: $VENV_PATH"
    python3.12 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install --upgrade pip -q

if python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch already installed${NC}"
else
    echo "Installing PyTorch nightly ROCm 7.1..."
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1
    echo -e "${GREEN}✓ PyTorch installed${NC}"
fi

# Step 4: Unsloth
echo ""
echo -e "${YELLOW}[4/4] Unsloth${NC}"
if python -c "import unsloth" 2>/dev/null; then
    echo -e "${GREEN}✓ Unsloth already installed${NC}"
else
    echo "Installing Unsloth..."
    pip install numpy transformers datasets accelerate peft trl huggingface_hub sentencepiece protobuf safetensors -q
    pip install --no-deps unsloth unsloth-zoo -q
    pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth" -q
    pip install cut_cross_entropy msgspec triton -q
    pip install torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1 --no-deps -q 2>/dev/null || true
    echo -e "${GREEN}✓ Unsloth installed${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Activate with:  source $VENV_PATH/bin/activate"
echo ""
if [[ "$(uname -r)" != "$KERNEL" ]]; then
    echo -e "${YELLOW}⚠ REBOOT REQUIRED to use kernel $KERNEL${NC}"
fi
