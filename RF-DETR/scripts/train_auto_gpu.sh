#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

if [ "${CUDA_VISIBLE_DEVICES+x}" = "x" ]; then
  VISIBLE_DEVICES=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr -d '[:space:]')
  if [ "${VISIBLE_DEVICES}" = "-1" ] || [ -z "${VISIBLE_DEVICES}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' 没有可用 GPU。" >&2
    exit 1
  fi
  NUM_GPUS=$(printf '%s' "${VISIBLE_DEVICES}" | awk -F',' '{count=0; for (i=1; i<=NF; i++) if ($i != "") count++; print count}')
else
  NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
  if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "ERROR: 未设置 CUDA_VISIBLE_DEVICES，且当前环境没有可见 GPU。" >&2
    exit 1
  fi
  VISIBLE_DEVICES="all-visible"
fi

if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "ERROR: 没有解析到可用 GPU。" >&2
  exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} (${NUM_GPUS} GPU(s))"

BASE_CMD="RF-DETR/scripts/train.py \
  --model-size seg-large \
  --work-dir RF-DETR/runs/rfdetr_seg_large_plain \
  --data-dir RF-DETR/data/plain \
  --raw-json dataset/trainval/trainval.json \
  --image-root dataset/trainval \
  --device cuda \
  --batch-size auto \
  --max-epochs 120"

if [ "${NUM_GPUS}" -eq 1 ]; then
  # shellcheck disable=SC2086
  exec python ${BASE_CMD} "$@"
fi

# RF-DETR 1.7.1 uses PyTorch Lightning internally. The script exposes all GPUs
# to Lightning rather than wrapping the Python entrypoint with torchrun.
# shellcheck disable=SC2086
exec python ${BASE_CMD} "$@"
