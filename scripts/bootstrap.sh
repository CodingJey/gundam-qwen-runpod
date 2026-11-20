#!/usr/bin/env bash
# Runtime bootstrap for Runpod's public ComfyUI base image.
#
# This is the no-GHCR-build fallback: deploy runpod/comfyui:latest with
#   entrypoint: ["/bin/bash", "-lc"]
#   cmd: ["curl -fsSL <this file> | bash"]
# It installs the extra system packages, ComfyUI-GGUF, and then hands off to
# the same /opt/gundam/start.sh used by the custom image.

set -euo pipefail

REPO_URL="${GUNDAM_REPO_URL:-https://github.com/CodingJey/gundam-qwen-runpod.git}"
SRC="/opt/gundam-src"
DST="/opt/gundam"
GGUF_SHA="6ea2651e7df66d7585f6ffee804b20e92fb38b8a"

export DEBIAN_FRONTEND=noninteractive

echo "[bootstrap] installing nginx, aria2, and htpasswd"
apt-get update
apt-get install -y --no-install-recommends nginx apache2-utils aria2 ca-certificates
rm -f /etc/nginx/sites-enabled/default

echo "[bootstrap] fetching project from ${REPO_URL}"
rm -rf "${SRC}"
git clone --depth 1 "${REPO_URL}" "${SRC}"

mkdir -p "${DST}/input" "${DST}/workflows"
cp "${SRC}/docker/extra_model_paths.yaml" "${DST}/"
cp "${SRC}/docker/nginx.conf" /etc/nginx/conf.d/gundam.conf
cp "${SRC}/config/models.json" "${DST}/"
cp "${SRC}"/workflows/*.json "${DST}/workflows/"
cp "${SRC}/assets/gundam_upload_placeholder.png" "${DST}/input/"
cp "${SRC}/scripts/download_models.py" "${DST}/"
cp "${SRC}/scripts/start.sh" "${DST}/"
chmod +x "${DST}/start.sh"

echo "[bootstrap] installing ComfyUI-GGUF"
GGUF_DIR="/opt/comfyui-baked/custom_nodes/ComfyUI-GGUF"
rm -rf "${GGUF_DIR}"
git clone https://github.com/city96/ComfyUI-GGUF.git "${GGUF_DIR}"
cd "${GGUF_DIR}"
git checkout "${GGUF_SHA}"
rm -rf .git
python3 -m pip install --no-cache-dir "gguf==0.17.1"

touch /etc/nginx/.htpasswd
chmod 644 /etc/nginx/.htpasswd

echo "[bootstrap] handing off to /opt/gundam/start.sh"
exec "${DST}/start.sh"
