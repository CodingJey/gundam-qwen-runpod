# Pinned Runpod ComfyUI base (linux/amd64). The base already contains ComfyUI,
# PyTorch, CUDA, SSH, and the usual Runpod helper scripts.
FROM runpod/comfyui:latest@sha256:7ed65713a24c7c836cea8e5ba338ee140b0257b0a086cc3835784e0473efe2f2

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HUB_DISABLE_TELEMETRY=1

# nginx + htpasswd provide a small authenticated reverse proxy in front of
# ComfyUI. tini gives us correct signal handling for background processes.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apache2-utils \
        aria2 \
        ca-certificates \
        curl \
        git \
        nginx \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

# ComfyUI-GGUF is required for Qwen-Image GGUF quantizations. Pinned so image
# rebuilds are reproducible.
ARG COMFYUI_GGUF_SHA=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
RUN git clone https://github.com/city96/ComfyUI-GGUF.git \
        /opt/comfyui-baked/custom_nodes/ComfyUI-GGUF && \
    cd /opt/comfyui-baked/custom_nodes/ComfyUI-GGUF && \
    git checkout "${COMFYUI_GGUF_SHA}" && \
    rm -rf .git && \
    python3 -m pip install --no-cache-dir "gguf==0.17.1"

# Project files
COPY docker/extra_model_paths.yaml /opt/gundam/extra_model_paths.yaml
COPY docker/nginx.conf /etc/nginx/conf.d/gundam.conf
COPY config/models.json /opt/gundam/models.json
COPY workflows/ /opt/gundam/workflows/
COPY assets/gundam_upload_placeholder.png /opt/gundam/input/gundam_upload_placeholder.png
COPY scripts/download_models.py /opt/gundam/download_models.py
COPY scripts/start.sh /opt/gundam/start.sh

RUN chmod +x /opt/gundam/start.sh && \
    python3 -m py_compile /opt/gundam/download_models.py && \
    touch /etc/nginx/.htpasswd && \
    chmod 644 /etc/nginx/.htpasswd && \
    nginx -t

WORKDIR /opt/comfyui-baked

EXPOSE 8188 22

# Runpod ignores Docker HEALTHCHECKs, but this is useful when testing locally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PUBLIC_PORT:-8188}/healthz" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/gundam/start.sh"]
