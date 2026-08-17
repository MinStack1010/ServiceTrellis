#!/usr/bin/env bash
# Runs on the GPU VM. Cloud Build uploads and invokes this script via IAP.
# All Python/CUDA/TRELLIS dependencies stay inside the Docker image.
set -euo pipefail

IMAGE="${1:?Image reference is required}"
HF_SECRET="${2:-trellis2-hf-token}"
REGISTRY="${IMAGE%%/*}"
CONTAINER="trellis2-api"

gcloud auth configure-docker "${REGISTRY}" --quiet
HF_TOKEN="$(gcloud secrets versions access latest --secret="${HF_SECRET}")"

docker pull "${IMAGE}"
NEW_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
RUNNING_IMAGE_ID="$(docker inspect --format '{{.Image}}' "${CONTAINER}" 2>/dev/null || true)"

if [[ "${NEW_IMAGE_ID}" == "${RUNNING_IMAGE_ID}" ]]; then
  echo "${CONTAINER} already runs this image."
  exit 0
fi

docker rm -f "${CONTAINER}" 2>/dev/null || true
docker run -d \
  --name "${CONTAINER}" \
  --restart unless-stopped \
  --gpus all \
  -p 127.0.0.1:8080:8080 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/data/huggingface \
  -v trellis-hf:/data/huggingface \
  "${IMAGE}"

unset HF_TOKEN
echo "${CONTAINER} is now running ${IMAGE}."
