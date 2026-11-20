#!/usr/bin/env bash
# Container entrypoint for the Gundam Qwen-Image pod.
#
#   Runpod HTTP proxy (HTTPS) -> nginx :8188 (Basic auth) -> ComfyUI :8189
#
# We deliberately keep using Runpod's own /start.sh for the Python venv,
# Jupyter/FileBrowser, and SSH setup. That script is battle-tested against the
# base image. This wrapper downloads models first, writes the ComfyUI args it
# should use, and puts nginx in front of ComfyUI.

set -uo pipefail
umask 022

log() {
    printf '[gundam] %s %s\n' "$(date -u +%H:%M:%S)" "$*"
}

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
COMFY_PORT="${COMFY_PORT:-8189}"
PUBLIC_PORT="${PUBLIC_PORT:-8188}"
WORKSPACE_COMFY="/workspace/runpod-slim/ComfyUI"
ARGS_FILE="/workspace/runpod-slim/comfyui_args.txt"

export MODELS_DIR
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p \
    "${MODELS_DIR}"/{diffusion_models,text_encoders,vae,loras,unet,checkpoints,controlnet,clip_vision,style_models,embeddings,upscale_models} \
    /workspace/{output,input,user,temp} \
    /workspace/runpod-slim

if [ -f /opt/gundam/input/gundam_upload_placeholder.png ] && \
   [ ! -f /workspace/input/gundam_upload_placeholder.png ]; then
    cp /opt/gundam/input/gundam_upload_placeholder.png /workspace/input/
fi

log "starting Gundam Qwen-Image container"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

# ---------------------------------------------------------------------------
# Models (first boot downloads ~20 GB; later boots reuse /workspace)
# ---------------------------------------------------------------------------
if [ "${SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    log "SKIP_MODEL_DOWNLOAD=1; not downloading models"
else
    log "checking model weights (first boot can take 10-20 minutes)"
    if ! python3 /opt/gundam/download_models.py; then
        log "model download reported an error; starting ComfyUI anyway"
    fi
fi

# ---------------------------------------------------------------------------
# Keep ComfyUI-GGUF present when /workspace already has an older ComfyUI copy.
# On a fresh workspace /start.sh copies the whole baked bundle for us.
# ---------------------------------------------------------------------------
if [ -d "${WORKSPACE_COMFY}" ]; then
    mkdir -p "${WORKSPACE_COMFY}/custom_nodes"
    rsync -a --delete \
        /opt/comfyui-baked/custom_nodes/ComfyUI-GGUF/ \
        "${WORKSPACE_COMFY}/custom_nodes/ComfyUI-GGUF/" || \
        log "WARNING: could not refresh ComfyUI-GGUF in the workspace"
fi

# ---------------------------------------------------------------------------
# Workflows: copy defaults without overwriting the user's edits.
# ---------------------------------------------------------------------------
WORKFLOW_DIR="/workspace/user/default/workflows"
mkdir -p "${WORKFLOW_DIR}"
for workflow in /opt/gundam/workflows/*.json; do
    [ -e "${workflow}" ] || continue
    destination="${WORKFLOW_DIR}/$(basename "${workflow}")"
    if [ ! -e "${destination}" ]; then
        cp "${workflow}" "${destination}"
        log "installed workflow: $(basename "${workflow}")"
    fi
done

STYLE_LORA_FILE="${GUNDAM_LORA_FILENAME:-gundam_qwen_image.safetensors}"
if [ -f "${MODELS_DIR}/loras/${STYLE_LORA_FILE}" ]; then
    log "Gundam style LoRA found: ${STYLE_LORA_FILE}"
    if [ "${STYLE_LORA_FILE}" = "gundam_qwen_image.safetensors" ]; then
        log "use the 'gundam_qwen_image' workflow"
    else
        log "select this filename in the Gundam style LoRA node"
    fi
else
    log "no Gundam style LoRA found; use 'qwen_image_base' for now"
fi

# ---------------------------------------------------------------------------
# Tell Runpod's /start.sh how to launch ComfyUI. Duplicate command-line flags
# are resolved by ComfyUI's argparse with the last value winning, so these
# override /start.sh's fixed --listen 0.0.0.0 --port 8188.
# ---------------------------------------------------------------------------
{
    echo "# Managed by /opt/gundam/start.sh; use COMFYUI_EXTRA_ARGS for additions."
    echo "--listen 127.0.0.1"
    echo "--port ${COMFY_PORT}"
    echo "--extra-model-paths-config /opt/gundam/extra_model_paths.yaml"
    echo "--output-directory /workspace/output"
    echo "--input-directory /workspace/input"
    echo "--user-directory /workspace/user"
    echo "--temp-directory /workspace/temp"
    echo "--preview-method none"
    echo "--reserve-vram 0.5"
    if [ "${COMFYUI_CACHE_NONE:-0}" = "1" ]; then
        echo "--cache-none"
        log "ComfyUI cache disabled (COMFYUI_CACHE_NONE=1)"
    fi
    if [ -n "${COMFYUI_EXTRA_ARGS:-}" ]; then
        echo "${COMFYUI_EXTRA_ARGS}"
        log "extra ComfyUI args: ${COMFYUI_EXTRA_ARGS}"
    fi
} > "${ARGS_FILE}"

# ---------------------------------------------------------------------------
# Basic auth for the public Runpod HTTP proxy
# ---------------------------------------------------------------------------
BASIC_AUTH_USER="${BASIC_AUTH_USER:-gundam}"
BASIC_AUTH_PASSWORD="${BASIC_AUTH_PASSWORD:-}"

if [ -z "${BASIC_AUTH_PASSWORD}" ]; then
    BASIC_AUTH_PASSWORD="$(openssl rand -hex 16)"
    log "BASIC_AUTH_PASSWORD was not set; generated one: ${BASIC_AUTH_PASSWORD}"
fi

case "${BASIC_AUTH_USER}" in
    *:*) log "BASIC_AUTH_USER must not contain a colon"; exit 1 ;;
esac

if ! htpasswd -bc /etc/nginx/.htpasswd "${BASIC_AUTH_USER}" "${BASIC_AUTH_PASSWORD}" >/dev/null 2>&1; then
    log "failed to create /etc/nginx/.htpasswd"
    exit 1
fi
chmod 644 /etc/nginx/.htpasswd

if [ "${PUBLIC_PORT}" != "8188" ]; then
    sed -i "s/listen 8188;/listen ${PUBLIC_PORT};/" /etc/nginx/conf.d/gundam.conf
fi

if ! nginx -t; then
    log "nginx configuration test failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# Start nginx and Runpod's /start.sh (Jupyter, FileBrowser, SSH, ComfyUI).
# ---------------------------------------------------------------------------
nginx -g 'daemon off;' &
NGINX_PID=$!
sleep 1
if ! kill -0 "${NGINX_PID}" 2>/dev/null; then
    log "nginx failed to start"
    exit 1
fi

log "starting Runpod /start.sh (ComfyUI will listen on 127.0.0.1:${COMFY_PORT})"
/start.sh &
BASE_PID=$!

shutdown() {
    log "shutting down"
    kill -TERM "${BASE_PID}" "${NGINX_PID}" 2>/dev/null || true
    wait "${BASE_PID}" "${NGINX_PID}" 2>/dev/null || true
    exit 0
}
trap shutdown TERM INT

if [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "UI: https://${RUNPOD_POD_ID}-${PUBLIC_PORT}.proxy.runpod.net"
fi
log "waiting for ComfyUI (models load on first generation)"

while kill -0 "${BASE_PID}" 2>/dev/null; do
    if ! kill -0 "${NGINX_PID}" 2>/dev/null; then
        log "nginx exited unexpectedly"
        kill -TERM "${BASE_PID}" 2>/dev/null || true
        wait "${BASE_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

wait "${BASE_PID}"
STATUS=$?
log "/start.sh exited with status ${STATUS}"
kill -TERM "${NGINX_PID}" 2>/dev/null || true
wait "${NGINX_PID}" 2>/dev/null || true
exit "${STATUS}"
