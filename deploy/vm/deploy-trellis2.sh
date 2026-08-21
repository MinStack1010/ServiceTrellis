#!/usr/bin/env bash
# Runs on the GPU VM. Cloud Build uploads and invokes this script via IAP.
# All Python/CUDA/TRELLIS dependencies stay inside the Docker image.
# Use the VM metadata identity directly. This avoids a root/user gcloud
# credential mismatch when Cloud Build invokes this script through SSH.
set -euo pipefail

IMAGE="${1:?Image reference is required}"
REGISTRY="${IMAGE%%/*}"
CONTAINER="trellis2-api"
HF_SECRET="trellis2-hf-token"

metadata() {
  curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"
}

PROJECT_ID="$(metadata project/project-id)"
ACCESS_TOKEN="$(metadata instance/service-accounts/default/token | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')"
HF_TOKEN="$({
  curl -fsS \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://secretmanager.googleapis.com/v1/projects/${PROJECT_ID}/secrets/${HF_SECRET}/versions/latest:access" \
    | python3 -c 'import base64, json, sys; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode(), end="")'
})"

# ── Docker network (cho Redis ↔ API communication) ───────────────────────────
NETWORK="trellis-net"
if ! docker network inspect "${NETWORK}" >/dev/null 2>&1; then
  docker network create "${NETWORK}"
  echo "Created Docker network: ${NETWORK}"
fi

# ── Redis (job state persistence) ────────────────────────────────────────────
# Start a Redis container on the same Docker bridge network if not already
# running. Data is persisted to the "trellis-redis" named volume so job state
# survives container restarts and deployments.
# Redis is only reachable inside the VM — not exposed to the internet.
if ! docker inspect redis >/dev/null 2>&1; then
  echo "Starting Redis container..."
  docker run -d \
    --name redis \
    --restart unless-stopped \
    --network "${NETWORK}" \
    -v trellis-redis:/data \
    redis:7-alpine redis-server --appendonly yes
  echo "Redis started."
else
  # Đảm bảo Redis đã join đúng network (trường hợp tạo lại network)
  docker network connect "${NETWORK}" redis 2>/dev/null || true
  echo "Redis already running."
fi

# Artifact Registry accepts the VM's short-lived OAuth token. Refresh it at
# every deployment instead of storing a Docker credential on the VM.
printf '%s' "${ACCESS_TOKEN}" | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY}"

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
  --network "${NETWORK}" \
  -p 0.0.0.0:8080:8080 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/data/huggingface \
  -e REDIS_URL="redis://redis:6379/0" \
  -v trellis-hf:/data/huggingface \
  -v trellis-tmp:/app/tmp \
  "${IMAGE}"

unset ACCESS_TOKEN HF_TOKEN
echo "${CONTAINER} is now running ${IMAGE}."
