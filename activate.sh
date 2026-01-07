#!/bin/bash
# Activate AI environment for Strix Halo
# Usage: source activate.sh

# Find venv
VENV_PATH=""
for path in ~/strix-halo-unsloth ~/rocm7_unsloth ~/rocm7_test; do
    if [[ -d "$path" && -f "$path/bin/activate" ]]; then
        VENV_PATH="$path"
        break
    fi
done

if [[ -z "$VENV_PATH" ]]; then
    echo "No venv found. Run: ./scripts/setup.sh"
    return 1 2>/dev/null || exit 1
fi

source "$VENV_PATH/bin/activate"

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export AMD_LOG_LEVEL=0
export HIP_VISIBLE_DEVICES=0

echo "Strix Halo AI environment activated"
echo "  Python:  $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
