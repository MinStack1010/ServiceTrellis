FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    ATTN_BACKEND=sdpa \
    SPARSE_ATTN_BACKEND=sdpa \
    SPARSE_CONV_BACKEND=flex_gemm \
    TORCH_CUDA_ARCH_LIST=8.9 \
    CMAKE_CUDA_ARCHITECTURES=89 \
    MAX_JOBS=4 \
    HF_HOME=/data/huggingface \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git libgl1 libglib2.0-0 libjpeg-dev ninja-build \
        python3-numpy python3-pip \
        blender && \
    rm -rf /var/lib/apt/lists/*

# Ensure numpy is available to Blender's Python (3-layer fallback).
RUN set -ex; \
    if blender --background --python-expr "import numpy; print('[Dockerfile] numpy OK:', numpy.__version__)" 2>/dev/null; then \
        echo "Layer 1: numpy already works in Blender"; \
    else \
        echo "Layer 2: finding Blender Python and pip installing..."; \
        BLENDER_PY=$(blender --background --python-expr "import sys; print(sys.executable)" 2>/dev/null | tail -1 | tr -d '[:space:]'); \
        echo "Blender Python: $BLENDER_PY"; \
        "$BLENDER_PY" -m pip install numpy 2>/dev/null && \
        blender --background --python-expr "import numpy; print('[Dockerfile] numpy OK after pip:', numpy.__version__)" 2>/dev/null && exit 0; \
        echo "Layer 3: copying numpy from system Python..."; \
        SYSTEM_NUMPY=$(/usr/bin/python3 -c "import numpy, os; print(os.path.dirname(numpy.__file__))" 2>/dev/null); \
        DEST=$("$BLENDER_PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null); \
        echo "System numpy: $SYSTEM_NUMPY -> $DEST"; \
        cp -r "$SYSTEM_NUMPY" "$DEST/" && \
        blender --background --python-expr "import numpy; print('[Dockerfile] numpy OK after copy:', numpy.__version__)"; \
    fi

# Ensure Pillow is available to Blender's Python for WebP->PNG conversion
RUN BLENDER_PY=$(blender --background --python-expr "import sys; print(sys.executable)" 2>/dev/null | tail -1 | tr -d '[:space:]'); \
    echo "Blender Python: $BLENDER_PY"; \
    "$BLENDER_PY" -m pip install Pillow 2>/dev/null || \
    (SYSTEM_PILLOW=$(/usr/bin/python3 -c "import PIL, os; print(os.path.dirname(PIL.__file__))" 2>/dev/null) && \
     DEST=$("$BLENDER_PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null) && \
     echo "Copying Pillow from system: $SYSTEM_PILLOW -> $DEST" && \
     cp -r "$SYSTEM_PILLOW" "$DEST/"); \
    blender --background --python-expr "from PIL import Image; print('[Dockerfile] Pillow OK:', Image.__version__)" 2>/dev/null || \
    echo "WARNING: Pillow not available in Blender - WebP conversion may fail"

WORKDIR /app

COPY requirements-service.txt ./
RUN pip install --upgrade pip && pip install -r requirements-service.txt

COPY docker/install-extensions.sh ./docker/install-extensions.sh
COPY o-voxel ./o-voxel
RUN chmod +x docker/install-extensions.sh && docker/install-extensions.sh
RUN python -c "import torch; assert torch.version.cuda == '12.4'; print(torch.__version__)" && pip check

COPY . .

RUN mkdir -p /data/huggingface /app/tmp

EXPOSE 8080

CMD ["sh", "-c", "python api_server.py --host 0.0.0.0 --port ${PORT:-8080}"]
