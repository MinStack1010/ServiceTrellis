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
  -p 0.0.0.0:8080:8080 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/data/huggingface \
  -v trellis-hf:/data/huggingface \
  "${IMAGE}"

unset ACCESS_TOKEN HF_TOKEN
echo "${CONTAINER} is now running ${IMAGE}."
