#!/usr/bin/env bash
# The task itself: build an environment in the task's working directory, train,
# publish, then prove the published model loads by predicting from the Hub.
#
# Takes its settings from the environment train.py (next to configure.py) sets on the task:
#   HF_REPO        org/name to publish to; empty trains without publishing
#   HF_TOKEN       write token for HF_REPO
#   HF_PUBLIC      "true" creates the repository as public
#   HF_REVISION    branch to publish to, "main" unless this is a sweep run
#   TORCH_DEVICE   "cuda" fails unless training can use a GPU; "cpu" is the default
#   RUN_ID         recorded with the model
#   TRAIN_ARGS     extra arguments for train_model.py
#
# Its output is also pushed to the Hugging Face repository's "logs" branch.
#
# It sets NODE_REGION, NODE_ZONE and NODE_INSTANCE_TYPE for train_model.py from the
# instance metadata, so the model records where it was trained.

set -euo pipefail

UV_VERSION=0.13.7
TORCH_VERSION=2.14.1
PYTHON_VERSION=3.12

export UV_INSTALL_DIR="$PWD/.uv/bin"
export UV_CACHE_DIR="$PWD/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv/python"
export UV_NO_MODIFY_PATH=1
export HF_HOME="$PWD/.hf"

TORCH_DEVICE="${TORCH_DEVICE:-cpu}"

# The newest PyTorch build this node's driver supports. Read from the driver
# because a CUDA build needs a driver at least as new, and images differ.
torch_index() {
  if [[ "$TORCH_DEVICE" != "cuda" ]]; then
    echo cpu
    return
  fi
  local cuda
  # "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer ones.
  cuda="$(nvidia-smi | sed -n 's/.*CUDA[A-Za-z ]*Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
  read -r major minor <<<"$cuda"
  local version=$((${major:-0} * 100 + ${minor:-0}))
  if ((version >= 1302)); then echo cu132
  elif ((version >= 1300)); then echo cu130
  elif ((version >= 1206)); then echo cu126
  else
    echo "driver reports CUDA '${cuda}'; PyTorch ${TORCH_VERSION} needs 12.6 or newer" >&2
    return 1
  fi
}

run_all() {
  set -euo pipefail
  if [[ "$TORCH_DEVICE" == "cuda" ]]; then
    echo "=== GPU"
    if ! command -v nvidia-smi >/dev/null; then
      echo "TORCH_DEVICE=cuda, but this node has no nvidia-smi: is it a GPU instance with a driver?" >&2
      exit 3
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  fi
  TORCH_INDEX="$(torch_index)"

  # Where this node is, from the EC2 instance metadata service (IMDSv2), recorded
  # with the model. Empty off EC2, which is fine.
  imds() {
    local token
    token="$(curl -fsS -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null)" || return 0
    curl -fsS -m 2 -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true
  }
  export NODE_REGION="$(imds placement/region)"
  export NODE_ZONE="$(imds placement/availability-zone)"
  export NODE_INSTANCE_TYPE="$(imds instance-type)"
  echo "=== Node: ${NODE_INSTANCE_TYPE:-unknown type} in ${NODE_ZONE:-an unknown zone}"

  echo "=== Installing uv ${UV_VERSION} and Python ${PYTHON_VERSION}"
  curl --proto '=https' --tlsv1.2 -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$UV_INSTALL_DIR:$PATH"
  uv venv --python "$PYTHON_VERSION" .venv
  export VIRTUAL_ENV="$PWD/.venv"

  echo "=== Installing PyTorch ${TORCH_VERSION} (${TORCH_INDEX}) and dependencies"
  uv pip install --quiet "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
  uv pip install --quiet -r requirements.txt

  echo "=== Training"
  publish_args=()
  if [[ -n "${HF_REPO:-}" ]]; then
    publish_args+=(--hf-repo "$HF_REPO" --hf-revision "${HF_REVISION:-main}")
    [[ "${HF_PUBLIC:-}" == "true" ]] && publish_args+=(--public)
  fi
  # TRAIN_ARGS is built by train.py from numeric options and is split on purpose.
  # shellcheck disable=SC2086
  .venv/bin/python train_model.py --run-id "${RUN_ID:-local}" --device "$TORCH_DEVICE" "${publish_args[@]}" ${TRAIN_ARGS:-}

  echo "=== Checking the result with a prediction"
  .venv/bin/python predict.py --model "${HF_REPO:-model-output}" --revision "${HF_REVISION:-main}"

  echo "=== Done"
}

# Every run's output goes to the repository's "logs" branch, as logs/<run id>.txt,
# whether it succeeds or fails. main, which holds the model, is never touched.
# Hugging Face is the only storage the task uses, so this needs no bucket. Plain
# curl, because a failed run may never have installed Python.
upload_log() {
  [[ -n "${HF_REPO:-}" && -n "${HF_TOKEN:-}" ]] || return 0
  local api="https://huggingface.co/api/models/$HF_REPO" auth="Authorization: Bearer $HF_TOKEN"
  # The repository may not exist if the run failed before publishing; then there's nowhere to put it.
  curl -fsS -m 30 -H "$auth" "$api" >/dev/null || return 1
  # 409 means the branch already exists, which is fine.
  curl -sS -m 30 -o /dev/null -X POST -H "$auth" -H "Content-Type: application/json" -d '{}' "$api/branch/logs"
  {
    printf '{"key":"header","value":{"summary":"Log of %s (exit %s)"}}\n' "${RUN_ID:-local}" "$1"
    printf '{"key":"file","value":{"path":"logs/%s.txt","encoding":"base64","content":"%s"}}\n' \
      "${RUN_ID:-local}" "$(base64 -w0 "$LOG")"
  } | curl -fsS -m 60 -o /dev/null -X POST -H "$auth" -H "Content-Type: application/x-ndjson" \
      --data-binary @- "$api/commit/logs"
}

LOG="$PWD/run.log"
# The pipeline runs run_all with errexit on; its status is the task's.
set +e
run_all 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
set -e
if upload_log "$status"; then
  [[ -n "${HF_REPO:-}" ]] && echo "log: https://huggingface.co/$HF_REPO/blob/logs/logs/${RUN_ID:-local}.txt"
else
  echo "could not upload the log to Hugging Face" >&2
fi
exit "$status"
