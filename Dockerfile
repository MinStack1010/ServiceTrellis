# Linux/CUDA image for the TRELLIS.2 API service.
# Requires an NVIDIA GPU with 24 GB+ VRAM and NVIDIA Container Toolkit.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Cloud Build has no GPU. Compile CUDA extensions only for the VM's L4
# (Ada, SM 8.9) instead of asking PyTorch to inspect a visible GPU.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    SPARSE_CONV_BACKEND=none \
    TORCH_CUDA_ARCH_LIST=8.9 \
    CMAKE_CUDA_ARCHITECTURES=89 \
    MAX_JOBS=4 \
    HF_HOME=/data/huggingface \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git libgl1 libglib2.0-0 libjpeg-dev ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-service.txt ./
RUN pip install --upgrade pip && pip install -r requirements-service.txt

# Keep expensive CUDA-extension compilation in a stable layer. Normal edits to
# api_server.py or Cloud Build files below will reuse this layer from :latest.
COPY docker/install-extensions.sh ./docker/install-extensions.sh
COPY o-voxel ./o-voxel
RUN chmod +x docker/install-extensions.sh && docker/install-extensions.sh
RUN python -c "import torch; assert torch.version.cuda == '12.4'; print(torch.__version__)" && pip check

# Application code changes do not invalidate the cached dependency layers.
COPY . .

RUN mkdir -p /data/huggingface /app/tmp

EXPOSE 8080

CMD ["sh", "-c", "python api_server.py --host 0.0.0.0 --port ${PORT:-8080}"]
