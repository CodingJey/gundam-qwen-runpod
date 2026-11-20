# Gundam Qwen-Image on Runpod

A small, self-contained setup for generating Gundam images with **Qwen-Image + your LoRA** on a rented **RTX 3090**:

- A **GitHub Actions** workflow builds the container image and pushes it to **GHCR** using your GitHub account.
- The image contains ComfyUI, ComfyUI-GGUF, nginx, and SSH. It does **not** contain the 20 GB of model weights.
- On first boot the container downloads the Qwen-Image GGUF stack into `/workspace`, so stopping and restarting the pod keeps the weights.
- ComfyUI is exposed through Runpod's HTTPS proxy on port `8188`, protected by HTTP Basic auth.
- Four ready-made ComfyUI workflows are installed: text-to-image and img2img, with and without the Gundam style LoRA.
- `scripts/generate.py` can queue generations from your laptop through the same ComfyUI API.

The container wrapper keeps Runpod's own `/start.sh` (Python venv, SSH, ComfyUI) and writes `/workspace/runpod-slim/comfyui_args.txt` so ComfyUI listens on `127.0.0.1:8189`. nginx listens on `8188` and adds Basic auth.

```
Your laptop
  ├─ scripts/launch.py ──> Runpod REST API v2 ──> RTX 3090 Pod
  └─ scripts/generate.py ─> https://<pod-id>-8188.proxy.runpod.net
                                      │
                                      ▼
                            nginx :8188 (Basic auth)
                                      │
                                      ▼
                            ComfyUI :8189
                                      │
                                      ▼
                    /workspace/models (pod volume disk)
```

## Why GGUF on a 3090?

The RTX 3090 is Ampere (sm_86). It has 24 GB of VRAM but no native FP8 tensor cores. An FP8 checkpoint loads, but its weights are dequantized to BF16 for compute, which gives back much of the memory pressure. The working 24 GB route is a GGUF quantized transformer:

| File | Size | Purpose |
| - | -: | - |
| `qwen-image-Q4_K_M.gguf` | ~13.1 GB | Qwen-Image 20B diffusion transformer |
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | ~4.7 GB | Text encoder |
| `qwen_image_vae.safetensors` | ~0.24 GB | VAE |
| `Qwen-Image-Lightning-8steps-V1.0.safetensors` | ~1.6 GB | 8-step speed LoRA |
| your Gundam LoRA | ~0.1-2 GB | Style/character adapter |

That is roughly 20 GB on disk and leaves headroom on a 24 GB card. The image defaults to `1024x1024`; Qwen's native `1328x1328` is possible but heavier. If you OOM, use `Q4_K_S` or `Q3_K_M` instead.

## Prerequisites

- A [Runpod account](https://console.runpod.io/) with credits (the official ComfyUI tutorial recommends at least $10).
- A GitHub account.
- Python 3.10+ on your laptop for the launcher/generator (standard library only; no `pip install` needed).
- Optional: a Qwen-Image LoRA (`.safetensors`) and, for Civitai downloads, a Civitai API token.
- No local Docker needed; GitHub Actions builds the image.

## 1. Build the image with GitHub Actions

Create a repository, push this directory, and let `.github/workflows/build.yml` run. It builds `linux/amd64` and pushes:

```
ghcr.io/<github-user>/<repo>:latest
ghcr.io/<github-user>/<repo>:sha-<short-sha>
```

The base image is pinned by digest and ComfyUI-GGUF is pinned by commit, so rebuilds are reproducible.

If you prefer to build locally instead of using Actions:

```bash
docker build --platform linux/amd64 -t ghcr.io/<you>/<repo>:latest .
docker push ghcr.io/<you>/<repo>:latest
```

### Private vs public GHCR package

GHCR packages are private by default. Choose one:

**A. Keep it private (recommended).**

1. Create a GitHub personal access token with `read:packages` (classic) or `Packages: Read` (fine-grained).
2. In Runpod, open **Credentials → Container Registry Auth → Add Credential**.
3. Registry: `ghcr.io`; username: your GitHub username; password: the token.
4. Copy the credential **ID** into `.env` as `RUNPOD_REGISTRY_AUTH_ID`.

**B. Make the package public.**

In the GitHub package settings, change visibility to public. Then leave `RUNPOD_REGISTRY_AUTH_ID` empty. This is simpler but exposes the image; the image contains no secrets.

## 2. Configure `.env`

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env
```

At minimum:

```dotenv
RUNPOD_API_KEY=...                 # Credentials → API Keys
RUNPOD_IMAGE=ghcr.io/<you>/<repo>:latest
BASIC_AUTH_PASSWORD=...            # openssl rand -hex 24
```

The API key needs **Pods Read/Write** and **Catalog Read**; add **Volumes Read** if you use `RUNPOD_NETWORK_VOLUME_ID`. A key with **All** permissions also works.

Also add your Gundam LoRA if you have it:

```dotenv
GUNDAM_LORA_URL=https://.../your-lora.safetensors
GUNDAM_LORA_FILENAME=gundam_qwen_image.safetensors
CIVITAI_TOKEN=...                  # only if the URL is a Civitai download
```

By default the container downloads the LoRA to `/workspace/models/loras/gundam_qwen_image.safetensors`. If you leave the URL blank, everything still works with the base Qwen-Image workflows.

Runpod SSH keys are optional for the UI but useful for uploads and debugging. Add one under **Credentials → SSH Public Keys**. The launcher creates the pod with `startSsh: true`.

## 3. Launch the 3090 pod

```bash
python3 scripts/launch.py up
```

What it does:

1. Reads the catalog with `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY`.
2. Orders data centers that currently report an RTX 3090, then tries to create the pod in each until one is placed.
3. Requests one RTX 3090 with **32 GB host RAM** minimum, a **40 GB container disk**, and a **50 GB volume disk** at `/workspace`.
4. Exposes `8188/http` (UI) and `22/tcp` (SSH).
5. Polls the pod until `RUNNING`, then polls `https://<pod-id>-8188.proxy.runpod.net/healthz` until ComfyUI answers.
6. Saves the pod id, URL, and SSH command to `.runpod-state.json`.

First boot downloads ~20 GB, so expect **10-20 minutes** on a fast Runpod host (slower networks take longer; the launcher waits up to 45 minutes by default). Later starts reuse the volume and are much faster. The readiness check only succeeds after the download and ComfyUI startup, so the command waits until the UI is actually usable.

If the pod is placed but readiness times out, the pod is left running (and billing). Follow logs with:

```bash
python3 scripts/launch.py logs
```

### Manual deployment in the Runpod console

The launcher is optional. If you prefer the console, deploy the image with:

- **GPU:** RTX 3090 (24 GB), 1 GPU.
- **Container disk:** 40 GB.
- **Volume disk:** 50 GB mounted at `/workspace` (or a network volume of at least 50 GB).
- **HTTP port:** `8188`.
- **TCP port:** `22` (optional, for SSH).
- **Environment:** `BASIC_AUTH_USER=gundam`, `BASIC_AUTH_PASSWORD=<long-random-value>`, plus optional `GUNDAM_LORA_URL`, `CIVITAI_TOKEN`, `HF_TOKEN`.
- **Registry credential:** required if the GHCR package is private.

The same image entrypoint handles model download, nginx, and ComfyUI.

## 4. Open the UI

The launcher prints:

```
UI:       https://<pod-id>-8188.proxy.runpod.net
User:     gundam
Password: (BASIC_AUTH_PASSWORD from .env)
```

Open the URL and sign in with Basic auth. The password is the one in `.env`, not a separate ComfyUI login.

In ComfyUI:

- Open **Workflows** (left sidebar).
- Use **`qwen_image_base`** if you have not added a Gundam LoRA yet.
- Use **`gundam_qwen_image`** after `gundam_qwen_image.safetensors` is present in `/workspace/models/loras/`.
- Use **`qwen_image_img2img_base`** or **`gundam_qwen_img2img`** to refine an existing image.

### Default settings

| Setting | Fast draft | Final quality |
| - | - | - |
| Lightning 8-step LoRA | enabled | bypassed (`Ctrl+B`) |
| Steps | 8 | 20-30 |
| CFG | 1.0 | 3.5-4.5 |
| Sampler / scheduler | `euler` / `simple` | `euler` / `simple` or `res_multistep` |
| Size | 1024x1024 | 1328x1328 if VRAM allows |
| Gundam LoRA strength | 0.85 | 0.8-1.0 |

Tweak the prompt, seed, dimensions, LoRA strength, steps, and CFG directly in the graph. Queue again; ComfyUI keeps the models cached for the next run.

For img2img, upload an image in the `LoadImage` node. `denoise=0.62` keeps the composition; lower it to `0.4-0.5` for cleanup, raise it to `0.7-0.8` for a stronger restyle.

> The included img2img workflow is standard latent img2img. For instruction-based edits ("remove the background", "change the text on the shoulder"), Qwen-Image-Edit is a separate 20B model. See [Qwen-Image-Edit in ComfyUI](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit) if you want to add it later.

## 5. Generate from the command line

`scripts/generate.py` uses the ComfyUI API, so it generates on the same pod you see in the browser.

```bash
python3 scripts/generate.py
python3 scripts/generate.py \
  --prompt "RX-78-2 Gundam standing in a rainy battlefield at night, searchlights, cinematic" \
  --width 832 --height 1216 --seed 42
python3 scripts/generate.py --no-lightning --steps 24 --cfg 4.0
python3 scripts/generate.py --style-strength 0.7
python3 scripts/generate.py --no-style-lora
```

- The default prompt and negative prompt are in the script; pass `--prompt`/`--negative` to change them.
- Images are written to `outputs/gundam_<timestamp>_<seed>_<n>.png`.
- The API graph is saved to `outputs/last_prompt.json` for debugging.
- `--dry-run` prints the JSON graph without contacting the pod.
- It reads `COMFYUI_URL`/`BASIC_AUTH_*` from `.env`, or the URL saved by `launch.py`.

## 6. Networking and security

### HTTP proxy

Runpod exposes an HTTP port internally as:

```
https://<POD_ID>-<INTERNAL_PORT>.proxy.runpod.net
```

So the UI is `https://<pod-id>-8188.proxy.runpod.net`. The proxy terminates HTTPS and forwards plain HTTP to nginx inside the pod. nginx adds Basic auth and proxies to ComfyUI on loopback `127.0.0.1:8189`. ComfyUI itself is not exposed directly.

The official proxy chain is:

```
Browser → Cloudflare → Runpod load balancer → Pod
```

Important consequences from the [Runpod port docs](https://docs.runpod.io/pods/configuration/expose-ports):

- **100-second HTTP timeout**: a single HTTP request cannot stay open forever. The ComfyUI prompt API returns immediately; progress is streamed over websockets and the CLI polls `/history`, so long generations are fine.
- **HTTPS only** at the proxy.
- **Public accessibility**: anyone with the URL can reach the Basic auth prompt. Use a long password. For team use, consider creating a template and referencing a Runpod secret from its environment instead of passing a plain value.

The `/healthz` endpoint is intentionally unauthenticated and only proxies ComfyUI's `/system_stats`, so the launcher can detect readiness without credentials.

The base image also starts JupyterLab on `8888` and FileBrowser on `8080`. The pod only exposes `8188/http` and `22/tcp`, so those two stay private; do not add them to the exposed ports unless you also secure them.

### TCP / SSH

The pod also exposes `22/tcp`. Runpod injects your account's registered SSH public keys when `startSsh: true`. The launcher prints the direct command if a public mapping exists, otherwise the Runpod SSH proxy command.

- On **Community Cloud**, a public IP is not guaranteed, so direct SSH may not be available. The HTTP UI still works through the proxy.
- Use **Secure Cloud** (`RUNPOD_CLOUD=SECURE`) if you need reliable direct SSH/SCP.
- The Runpod SSH proxy is interactive-shell only; `scp`, `rsync`, and port forwarding need the direct mapping.

### Changing the password later

The password is a pod environment variable. Changing `.env` alone does not change a running pod. Either edit the pod's env in the Runpod console and restart it, or terminate and recreate the pod.

## 7. Storage, cost, and lifecycle

Prices below are from the [Runpod pricing page](https://www.runpod.io/pricing) (November 2026) and can change.

| Resource | Price |
| - | -: |
| RTX 3090 Community Cloud | ~$0.22/hr |
| RTX 3090 Secure Cloud | ~$0.50/hr |
| Container disk | $0.10/GB/month |
| Volume disk while running | $0.10/GB/month |
| Volume disk while stopped | $0.20/GB/month |
| Network volume (standard) | $0.07/GB/month |
| Network volume (high performance) | $0.14/GB/month |

The default 40 GB container disk + 50 GB volume disk is about **$9/month** of storage while running (90 GB at $0.10/GB-month) and about **$10/month** while stopped (the 50 GB volume at $0.20/GB-month; the container disk is not charged). A one-hour session is roughly **$0.24** on Community Cloud with the default storage. **Compute billing stops when the pod stops; volume storage billing does not.** Terminate the pod when you are done.

```bash
python3 scripts/launch.py stop       # keeps /workspace, stops GPU billing
python3 scripts/launch.py start      # resumes the same pod
python3 scripts/launch.py down --yes # terminates and deletes the volume disk
```

A stopped pod keeps its original container image. After pushing a new image tag, use `down --yes` and `up` to recreate the pod with the new image.

### Persistence choices

- **Default (volume disk)**: `/workspace` survives stop/restart, is deleted on terminate, and the pod can be placed in any data center. This is the simplest option.
- **Network volume**: `/workspace` survives pod deletion and can be moved between pods in the same data center. Create one in the Runpod console, set `RUNPOD_NETWORK_VOLUME_ID` in `.env`, and the launcher will pin the pod to that volume's data center. This is cheaper for long-term model storage.
- **Global volume**: region-independent model storage, but it is object-backed rather than fully POSIX and is not a good place for ComfyUI's Python environment. The official ComfyUI template warns against using one as its primary `/workspace`. The default volume disk or a network volume is the better fit here.

### Downloading outputs

Images are saved in ComfyUI under `/workspace/output` and are downloadable from the UI. You can also use `scp` with the direct SSH mapping, or attach a network volume and read it through Runpod's S3-compatible API.

## 8. Using your own Gundam LoRA

The container expects the style LoRA at:

```
/workspace/models/loras/gundam_qwen_image.safetensors
```

Three ways to get it there:

1. **Direct URL** (set `GUNDAM_LORA_URL` in `.env` before `up`). For Civitai, filter by base model **Qwen**, open the model version, copy its download URL, and also set `CIVITAI_TOKEN`; the download API returns `401` without it.
2. **`scp`** to the direct SSH mapping:
   ```bash
   scp -P <port> your_lora.safetensors \
     root@<host>:/workspace/models/loras/gundam_qwen_image.safetensors
   ```
3. **`runpodctl send` / receive** or the Runpod web terminal. The exact transfer method is less important than the destination path.

Then:

1. Refresh ComfyUI.
2. Load `gundam_qwen_image`.
3. Select the file in the **Gundam style LoRA** node if it is not already selected.
4. Add the LoRA's trigger words at the start of the positive prompt.
5. Start at strength `0.85`, then try `0.6-1.0`.

**Compatibility checks:**

- It must be a **Qwen-Image** LoRA, not SDXL, Flux, or Pony. Those will not load correctly.
- A LoRA trained for **Qwen-Image-Edit** is not the same as a LoRA for base Qwen-Image. Use the matching base model.
- GGUF + LoRA support in ComfyUI-GGUF is marked experimental. If your LoRA fails to apply, try bypassing the Lightning LoRA first, or switch to the FP8 model (at the cost of more VRAM pressure on Ampere).
- If you stack multiple LoRAs, keep their combined strength reasonable; `0.6-0.8` each is a good starting point.

If the filename differs, either rename the file or set `GUNDAM_LORA_FILENAME` in `.env` and update the node selection in the workflow (or use `scripts/generate.py --style-lora <filename>`).

## 9. Model files and changing the quantization

The download manifest is `config/models.json`. It downloads with `aria2c` (16 parallel connections) and skips files whose size already matches the known upstream size, so restarts do not re-download 20 GB. It downloads:

- `city96/Qwen-Image-gguf` → `qwen-image-Q4_K_M.gguf`
- `unsloth/Qwen2.5-VL-7B-Instruct-GGUF` → `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- `Comfy-Org/Qwen-Image_ComfyUI` → `split_files/vae/qwen_image_vae.safetensors`
- `lightx2v/Qwen-Image-Lightning` → `Qwen-Image-Lightning-8steps-V1.0.safetensors`

To use a different quant, change the diffusion entry in `config/models.json` (for example `qwen-image-Q5_K_M.gguf`), update the workflow's `UnetLoaderGGUF` widget, and pass `--model qwen-image-Q5_K_M.gguf` to `scripts/generate.py`. The text encoder can stay at `Q4_K_M`.

Rough 3090 guidance:

- `Q4_K_M` (default): best speed/VRAM balance.
- `Q5_K_M`: slightly better detail, less headroom.
- `Q6_K`: may fit but leaves little room for activations and LoRAs.
- `Q8_0`: same size as FP8; not the best choice on Ampere.
- `Q3_K_M`/`Q3_K_S`: use if you need more headroom or higher resolution.

Qwen-Image is Apache-2.0. The Lightning LoRA and community LoRAs have their own licenses; check the model pages.

## 10. Troubleshooting

**`502 Bad Gateway` / "Not Ready"**
The pod is running but ComfyUI has not started yet. First boot downloads ~20 GB. Check:
```bash
python3 scripts/launch.py logs
```
Wait for `[models] all required models are present` and `Starting ComfyUI with args:` in the logs.

**`401` from the UI or `generate.py`**
The Basic auth credentials are wrong. Check `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD` in `.env`. The password is set at pod creation; changing `.env` does not update a running pod.

**Image pull fails**
If the GHCR package is private, `RUNPOD_REGISTRY_AUTH_ID` must point to a Runpod Container Registry Auth credential for `ghcr.io` with a token that can read the package. If the package is public, leave it blank. Check the pod's system logs for `pull access denied`.

**No 3090 capacity**
The launcher tries every data center the catalog reports. If all fail:
```dotenv
RUNPOD_CLOUD=SECURE
# or
RUNPOD_REGION=EUROPE
# or
RUNPOD_GPU=NVIDIA GeForce RTX 4090
```
The 4090 is faster but more expensive; the same GGUF stack fits.

**Out of memory**
- Lower the resolution to `1024x1024` or `832x1216`.
- Use `Q4_K_S` or `Q3_K_M`.
- Bypass the style LoRA while testing.
- Set `COMFYUI_CACHE_NONE=1` in `.env` and recreate/restart the pod. This reduces RAM/VRAM caching at the cost of reloading models.
- Stop other workflows; ComfyUI keeps models cached between runs.

**LoRA has no effect**
- Confirm the file is a Qwen-Image LoRA and is selected in the `LoraLoaderModelOnly` node.
- Add the trigger words from the LoRA page.
- Raise strength to `0.9-1.0`.
- With GGUF, LoRA support is experimental; try disabling the Lightning LoRA to isolate the issue.

**`model not found` / empty dropdown**
The model download may have failed. Check:
```bash
python3 scripts/launch.py logs
```
Look for the failing filename and the HTTP error. Civitai needs `CIVITAI_TOKEN`; gated Hugging Face repos need `HF_TOKEN`. You can also restart the pod; the downloader resumes `.part` files.

**Pod is running but UI URL does not resolve**
Wait for the pod status to be `RUNNING`, then check `/healthz`. The proxy URL only works while the pod exists. `python3 scripts/launch.py status` shows the current state.

**First generation is slow**
The first run loads the transformer and text encoder from disk into VRAM. Later runs reuse them. With the 8-step Lightning LoRA, a 1024x1024 draft on a 3090 is typically much faster than a 20-30 step run.

**Where are the logs?**
```bash
python3 scripts/launch.py logs
# or, if you have runpodctl:
runpodctl pod logs <pod-id> --follow
```

## 11. File map

```
.
├── .github/workflows/build.yml     # GHCR build/push
├── Dockerfile                      # ComfyUI + GGUF + nginx + SSH
├── assets/                         # img2img placeholder input image
├── config/models.json              # model download manifest
├── docker/
│   ├── extra_model_paths.yaml      # /workspace/models -> ComfyUI
│   └── nginx.conf                  # Basic auth + websocket proxy
├── scripts/
│   ├── download_models.py          # resumable model downloader
│   ├── start.sh                    # container entrypoint
│   ├── launch.py                   # Runpod API v2 launcher
│   ├── generate.py                 # ComfyUI API client
│   └── gundam_common.py            # shared helpers
├── tools/make_workflows.py         # regenerates the workflow JSON files
└── workflows/
    ├── gundam_qwen_image.json
    ├── qwen_image_base.json
    ├── gundam_qwen_img2img.json
    └── qwen_image_img2img_base.json
```

Regenerate the workflow JSON after changing defaults:

```bash
python3 tools/make_workflows.py
```

## 12. Official documentation used

- [Runpod Pods overview](https://docs.runpod.io/pods/overview)
- [Expose ports (HTTP proxy and TCP)](https://docs.runpod.io/pods/configuration/expose-ports)
- [Generate images with ComfyUI on Runpod](https://docs.runpod.io/tutorials/pods/comfyui)
- [Runpod storage options](https://docs.runpod.io/pods/storage/types)
- [Runpod credentials (API keys, SSH keys, registry auth, secrets)](https://docs.runpod.io/get-started/credentials)
- [Create a Pod with REST API v2](https://docs.runpod.io/api-reference-v2/pods/create-a-pod)
- [Pod state transitions](https://docs.runpod.io/api-reference-v2/pods/trigger-a-pod-state-transition)
- [List data centers with GPU availability](https://docs.runpod.io/api-reference-v2/catalog/list-data-centers)
- [Runpod GPU types](https://docs.runpod.io/references/gpu-types)
- [Runpod Pod pricing](https://docs.runpod.io/pods/pricing)
- [ComfyUI Qwen-Image tutorial](https://docs.comfy.org/tutorials/image/qwen/qwen-image)
- [ComfyUI Qwen-Image-Edit tutorial](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

> Runpod is styled with one capital letter; older documentation may use "RunPod".
