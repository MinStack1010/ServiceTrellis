#!/bin/bash
# Chạy 1 LẦN DUY NHẤT trên máy local (đã cài gcloud, đã login).
#
#   gcloud auth login
#   gcloud config set project synode-395620
#   export PROJECT_ID=synode-395620
#   bash deploy/gcp-setup-once.sh
#
# Sau đó tạo Cloud Build Trigger trên Console (bước 5 in ra link).
# Rồi git push main → tự build + deploy → có URL API.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. export PROJECT_ID=synode-395620}"
REGION="${REGION:-us-central1}"
AR_LOCATION="${AR_LOCATION:-asia-southeast1}"
REPO_NAME="trellis2"
SERVICE_NAME="trellis2"
IMAGE="${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/service"

echo "Project:  $PROJECT_ID"
echo "Region:   $REGION (Cloud Run GPU)"
echo "Registry: $IMAGE"
echo ""

echo "=== 1/6 Bật API ==="
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project="$PROJECT_ID"

echo "=== 2/6 Artifact Registry ==="
gcloud artifacts repositories describe "$REPO_NAME" \
  --location="$AR_LOCATION" --project="$PROJECT_ID" 2>/dev/null \
|| gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$AR_LOCATION" \
  --project="$PROJECT_ID"

echo "=== 3/6 Secret HuggingFace token ==="
if ! gcloud secrets describe trellis2-hf-token --project="$PROJECT_ID" >/dev/null 2>&1; then
  read -rsp "Dán HF_TOKEN (Enter): " HF_TOKEN
  echo
  printf '%s' "$HF_TOKEN" | gcloud secrets create trellis2-hf-token \
    --data-file=- --project="$PROJECT_ID"
else
  echo "Secret trellis2-hf-token đã có — bỏ qua"
fi

echo "=== 4/6 Phân quyền Cloud Build ==="
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${CB_SA}" --role="$ROLE" --quiet >/dev/null
done

gcloud secrets add-iam-policy-binding trellis2-hf-token \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/secretmanager.secretAccessor" \
  --project="$PROJECT_ID" --quiet >/dev/null

gcloud secrets add-iam-policy-binding trellis2-hf-token \
  --member="serviceAccount:${RUN_SA}" \
  --role="roles/secretmanager.secretAccessor" \
  --project="$PROJECT_ID" --quiet >/dev/null

echo "=== 5/6 Kiểm tra quota GPU (Cloud Run) ==="
echo "Cần bật billing + quota GPU L4 tại region ${REGION}."
echo "Console: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
echo ""

echo "=== 6/6 Tạo Cloud Build Trigger (Console, 1 lần) ==="
echo ""
echo "Mở: https://console.cloud.google.com/cloud-build/triggers/add?project=${PROJECT_ID}"
echo ""
echo "  1. Connect repository → chọn GitHub repo của bạn"
echo "  2. Event: Push to branch ^main\$"
echo "  3. Nếu repo là monorepo (có folder TRELLIS.2/):"
echo "       Configuration = Cloud Build configuration file"
echo "       Location      = Repository"
echo "       File          = TRELLIS.2/cloudbuild.yaml"
echo "     Nếu repo CHỈ có nội dung TRELLIS.2:"
echo "       File          = cloudbuild.yaml"
echo "  4. Substitution variables:"
echo "       _IMAGE        = ${IMAGE}"
echo "       _REGION       = ${REGION}"
echo "       _SERVICE_NAME = ${SERVICE_NAME}"
echo "       _DEPLOY       = true"
echo "  5. Save"
echo ""
echo "=== Xong setup ==="
echo "Push code:  git push origin main"
echo "Xem build:  https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}"
echo "Lấy URL API sau khi deploy xong:"
echo "  gcloud run services describe ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID} --format='value(status.url)'"
