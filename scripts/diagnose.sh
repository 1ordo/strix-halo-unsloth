#!/bin/bash
# Quick diagnostic for Strix Halo GPU setup
# Usage: ./scripts/diagnose.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════"
echo "  Strix Halo (gfx1151) Diagnostic"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Kernel check
KERNEL=$(uname -r)
echo -n "Kernel: $KERNEL "
if [[ "$KERNEL" == "6.17.8-300.fc43.x86_64" ]]; then
    echo -e "${GREEN}✓${NC}"
elif [[ "$KERNEL" == 6.17.9* ]] || [[ "$KERNEL" == 6.18* ]] || [[ "$KERNEL" == 6.19* ]]; then
    echo -e "${RED}✗ BROKEN - use 6.17.8${NC}"
else
    echo -e "${YELLOW}⚠ untested${NC}"
fi

# Boot params
echo ""
echo "Boot parameters:"
CMDLINE=$(cat /proc/cmdline)
for param in "amdgpu.cwsr_enable=0" "amd_iommu=off" "amdgpu.gttsize=131072"; do
    if echo "$CMDLINE" | grep -q "$param"; then
        echo -e "  ${GREEN}✓${NC} $param"
    else
        echo -e "  ${RED}✗${NC} $param ${RED}MISSING${NC}"
    fi
done

# GPU detection
echo ""
echo -n "GPU: "
if command -v rocminfo &> /dev/null; then
    GPU=$(rocminfo 2>/dev/null | grep -oP "gfx\d+" | head -1)
    if [[ "$GPU" == "gfx1151" ]]; then
        echo -e "${GREEN}$GPU ✓${NC}"
    elif [[ -n "$GPU" ]]; then
        echo -e "${YELLOW}$GPU${NC}"
    else
        echo -e "${RED}not detected${NC}"
    fi
else
    echo -e "${YELLOW}rocminfo not installed${NC}"
fi

# VRAM
echo -n "VRAM: "
if command -v rocm-smi &> /dev/null; then
    VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total" | awk '{print $3}')
    if [[ -n "$VRAM" ]]; then
        echo -e "${GREEN}${VRAM}${NC}"
    else
        echo -e "${YELLOW}unknown${NC}"
    fi
else
    echo -e "${YELLOW}rocm-smi not installed${NC}"
fi

# PyTorch
echo ""
echo -n "PyTorch: "
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [[ "$CUDA" == "True" ]]; then
        echo -e "  CUDA/ROCm: ${GREEN}available ✓${NC}"
    else
        echo -e "  CUDA/ROCm: ${RED}not available${NC}"
    fi
else
    echo -e "${YELLOW}not installed${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
