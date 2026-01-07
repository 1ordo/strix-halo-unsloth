# Strix Halo AI Toolbox
# PyTorch ROCm 7.1 + Unsloth for AMD Strix Halo (gfx1151)
#
# Build: podman build -t strix-halo-unsloth -f Containerfile .
# Run:   podman run -it --rm \
#            --device /dev/kfd \
#            --device /dev/dri \
#            --security-opt seccomp=unconfined \
#            --group-add video \
#            --group-add render \
#            -v ~/models:/models \
#            strix-halo-unsloth
#
# Inspired by kyuz0's amd-strix-halo-toolboxes:
# https://github.com/kyuz0/amd-strix-halo-toolboxes

FROM fedora:43

LABEL maintainer="1ordo"
LABEL description="Unsloth + PyTorch ROCm 7.1 for AMD Strix Halo (gfx1151)"
LABEL version="1.0"

# Install system dependencies
RUN dnf update -y && \
    dnf install -y \
        python3 \
        python3-pip \
        python3-devel \
        git \
        wget \
        curl \
        vim \
        htop \
        rocminfo \
        rocm-smi \
        --skip-unavailable \
        && \
    dnf clean all

# Python is already default in Fedora 43

# Create app user and directories
RUN useradd -m -s /bin/bash unsloth && \
    mkdir -p /models /datasets /workspace && \
    chown -R unsloth:unsloth /models /datasets /workspace

# Switch to app user
USER unsloth
WORKDIR /home/unsloth

# Create virtual environment
RUN python3 -m venv /home/unsloth/venv

# Activate venv for all subsequent commands
ENV PATH="/home/unsloth/venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/unsloth/venv"

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with ROCm 7.1 nightly
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.1

# Install Unsloth dependencies
RUN pip install \
    numpy \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    huggingface_hub \
    sentencepiece \
    protobuf \
    safetensors

# Install Unsloth AMD branch
RUN pip install --no-deps unsloth unsloth-zoo && \
    pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"

# Install additional dependencies
RUN pip install cut_cross_entropy msgspec triton

# Install torchvision (no deps to avoid torch conflict)
RUN pip install torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1 --no-deps || true

# Environment variables for AMD ROCm
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV AMD_LOG_LEVEL=0
ENV HIP_VISIBLE_DEVICES=0

# Suppress libdrm warning (cosmetic)
ENV LD_LIBRARY_PATH="/usr/lib64:$LD_LIBRARY_PATH"

# Volumes
VOLUME ["/models", "/datasets", "/workspace"]

# Working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1
