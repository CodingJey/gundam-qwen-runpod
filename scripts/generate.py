#!/usr/bin/env python3
"""Queue a Gundam Qwen-Image generation through ComfyUI's HTTP API.

The script talks to the same ComfyUI instance you open in the browser, so it
uses the Basic auth credentials from .env and the pod URL saved by launch.py.

Examples:
    python3 scripts/generate.py
    python3 scripts/generate.py --prompt "RX-78-2 Gundam on a rainy battlefield" \
        --width 832 --height 1216 --seed 42
    python3 scripts/generate.py --no-lightning --steps 24 --cfg 4.0
    python3 scripts/generate.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from gundam_common import (
    HttpError,
    ROOT,
    basic_auth_header,
    format_duration,
    read_state,
    request_json,
    setting,
)

DEFAULT_PROMPT = (
    "Mobile Suit Gundam mecha, white and blue armor with red and yellow accents, "
    "glowing green camera eyes, V-fin antenna, precise panel lines and decals, "
    "battle damage and weathering, standing in a ruined space colony hangar, "
    "sparks, smoke, volumetric light, cinematic low-angle shot, realistic "
    "sci-fi concept art, incredibly detailed, sharp focus, masterpiece"
)
DEFAULT_NEGATIVE = (
    "low quality, blurry, deformed, extra limbs, bad anatomy, watermark, "
    "signature, jpeg artifacts, toy, plastic"
)

DEFAULT_MODEL = "qwen-image-Q4_K_M.gguf"
DEFAULT_CLIP = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
DEFAULT_VAE = "qwen_image_vae.safetensors"
DEFAULT_SPEED_LORA = "Qwen-Image-Lightning-8steps-V1.0.safetensors"
DEFAULT_STYLE_LORA = "gundam_qwen_image.safetensors"


def log(message: str) -> None:
    print(f"[generate] {message}", flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"[generate] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def comfy_settings(args: argparse.Namespace) -> dict[str, str]:
    state = read_state()
    base_url = (args.url or setting("COMFYUI_URL") or state.get("url") or "").rstrip("/")
    if not base_url:
        die(
            "no ComfyUI URL. Run 'python3 scripts/launch.py up' first, or pass --url "
            "https://<pod-id>-8188.proxy.runpod.net."
        )
    user = args.user or setting("COMFYUI_USER", setting("BASIC_AUTH_USER", "gundam"))
    password = args.password or setting("COMFYUI_PASSWORD") or setting("BASIC_AUTH_PASSWORD")
    if not password:
        die("no Basic auth password. Set BASIC_AUTH_PASSWORD in .env or pass --password.")
    return {"url": base_url, "user": user, "password": password}


def comfy_request(
    settings: dict[str, str],
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
) -> Any:
    headers = {
        "Authorization": basic_auth_header(settings["user"], settings["password"]),
    }
    return request_json(
        method,
        settings["url"] + path,
        body=body,
        headers=headers,
        timeout=timeout,
        retries=2,
    )


def describe_server(settings: dict[str, str]) -> None:
    try:
        stats = comfy_request(settings, "GET", "/system_stats", timeout=20)
    except HttpError as exc:
        if exc.status == 401:
            die(
                "ComfyUI rejected the Basic auth credentials. Check BASIC_AUTH_USER "
                "and BASIC_AUTH_PASSWORD in .env."
            )
        if exc.status in {502, 503, 504}:
            die("ComfyUI is not ready yet (the Runpod proxy returned 502/503). Wait for the first boot/model download to finish.")
        die(f"could not reach ComfyUI: {exc}")
    except RuntimeError as exc:
        die(f"could not reach ComfyUI: {exc}")

    devices = (stats or {}).get("devices") or []
    for device in devices:
        name = device.get("name", "GPU")
        vram = device.get("vram_total")
        if isinstance(vram, (int, float)) and vram > 0:
            log(f"server: {name} ({vram / 1024**3:.1f} GiB VRAM)")
        else:
            log(f"server: {name}")
    version = (stats or {}).get("system", {}).get("comfyui_version")
    if version:
        log(f"ComfyUI version: {version}")


def node_info(settings: dict[str, str], class_type: str) -> dict[str, Any]:
    try:
        response = comfy_request(settings, "GET", f"/object_info/{class_type}", timeout=60)
    except HttpError as exc:
        if exc.status == 404:
            die(
                f"ComfyUI does not expose node {class_type}. "
                "The custom node may have failed to load; check the pod logs."
            )
        die(f"could not read node info for {class_type}: {exc}")
    except RuntimeError as exc:
        die(f"could not read node info for {class_type}: {exc}")

    if not response or class_type not in response:
        die(
            f"ComfyUI does not expose node {class_type}. "
            "ComfyUI-GGUF may have failed to load; check the pod logs."
        )
    return response[class_type]


def combo_options(info: dict[str, Any], input_name: str) -> list[str]:
    inputs = info.get("input") or {}
    spec = (inputs.get("required") or {}).get(input_name)
    if spec is None:
        spec = (inputs.get("optional") or {}).get(input_name)
    if not spec:
        return []
    first = spec[0]
    if isinstance(first, (list, tuple)):
        return [str(value) for value in first]
    return []


def ensure_available(
    label: str,
    value: str,
    options: list[str],
    required: bool = True,
) -> bool:
    if not options:
        message = (
            f"{label} has no available files. The model may still be downloading, "
            "or the extra_model_paths.yaml mapping is wrong."
        )
        if required:
            die(message)
        log("WARNING: " + message)
        return False
    if value in options:
        return True

    message = f"{label} '{value}' is not available in ComfyUI."
    if required:
        die(message + " Available: " + ", ".join(options[:20]))
    log("WARNING: " + message + " Skipping it.")
    return False


def build_graph(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    model: str,
    clip: str,
    vae: str,
    speed_lora: str | None,
    style_lora: str | None,
    style_strength: float,
) -> dict[str, Any]:
    graph: dict[str, Any] = {}
    graph["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model}}
    model_ref: list[Any] = ["1", 0]

    if speed_lora:
        graph["2"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": speed_lora,
                "strength_model": 1.0,
            },
        }
        model_ref = ["2", 0]

    if style_lora:
        graph["3"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": style_lora,
                "strength_model": style_strength,
            },
        }
        model_ref = ["3", 0]

    graph["4"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": model_ref, "shift": 3.1},
    }
    graph["5"] = {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {"clip_name": clip, "type": "qwen_image"},
    }
    graph["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["5", 0], "text": prompt},
    }
    graph["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["5", 0], "text": negative},
    }
    graph["8"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    graph["9"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["8", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
        },
    }
    graph["10"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
    graph["11"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["10", 0]},
    }
    graph["12"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["11", 0], "filename_prefix": "gundam"},
    }
    return graph


def queue_prompt(settings: dict[str, str], graph: dict[str, Any]) -> str:
    client_id = f"gundam-{random.randint(0, 2**32 - 1):08x}"
    try:
        response = comfy_request(
            settings,
            "POST",
            "/prompt",
            {"prompt": graph, "client_id": client_id},
            timeout=60,
        )
    except HttpError as exc:
        if exc.status == 400:
            die(
                "ComfyUI rejected the workflow:\n"
                + json.dumps(exc.body, indent=2, ensure_ascii=False)
            )
        die(f"could not queue the prompt: {exc}")
    except RuntimeError as exc:
        die(f"could not queue the prompt: {exc}")

    if not isinstance(response, dict) or not response.get("prompt_id"):
        die(f"unexpected /prompt response: {response!r}")
    return str(response["prompt_id"])


def has_execution_error(entry: dict[str, Any]) -> bool:
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        return True
    for message in status.get("messages") or []:
        if isinstance(message, list) and message and message[0] == "execution_error":
            return True
    return False


def wait_for_result(
    settings: dict[str, str],
    prompt_id: str,
    timeout: int,
) -> dict[str, Any] | None:
    started = time.monotonic()
    last_note = 0.0
    while time.monotonic() - started < timeout:
        try:
            history = comfy_request(
                settings,
                "GET",
                f"/history/{prompt_id}",
                timeout=30,
            )
        except (HttpError, RuntimeError) as exc:
            log(f"history check failed ({exc}); retrying")
            time.sleep(3)
            continue

        entry = (history or {}).get(prompt_id)
        if entry:
            if has_execution_error(entry):
                status = entry.get("status") or {}
                print(
                    json.dumps(status, indent=2, ensure_ascii=False),
                    file=sys.stderr,
                )
                return None
            if (entry.get("status") or {}).get("completed") or entry.get("outputs"):
                return entry

        now = time.monotonic()
        if now - last_note >= 15:
            log(f"generating... {format_duration(now - started)} elapsed")
            last_note = now
        time.sleep(2)

    log(f"timed out after {format_duration(timeout)}")
    return None


def download_output(
    settings: dict[str, str],
    image: dict[str, Any],
    output_dir: Path,
    seed: int,
    index: int,
) -> Path:
    query = urlencode(
        {
            "filename": image.get("filename", "output.png"),
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
    )
    request = Request(
        settings["url"] + "/view?" + query,
        headers={
            "Authorization": basic_auth_header(settings["user"], settings["password"]),
            "User-Agent": "gundam-qwen-runpod/1.0",
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(image.get("filename", "output.png"))).suffix or ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_dir / f"gundam_{timestamp}_{seed}_{index}{suffix}"

    try:
        with urlopen(request, timeout=300) as response:
            destination.write_bytes(response.read())
    except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
        die(f"could not download generated image: {exc}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--speed-lora", default=DEFAULT_SPEED_LORA)
    parser.add_argument(
        "--style-lora",
        default=(
            setting("GUNDAM_LORA_NAME")
            or setting("GUNDAM_LORA_FILENAME")
            or DEFAULT_STYLE_LORA
        ),
    )
    parser.add_argument("--style-strength", type=float, default=0.85)
    parser.add_argument("--no-lightning", action="store_true", help="disable the 8-step speed LoRA")
    parser.add_argument("--no-style-lora", action="store_true", help="disable the Gundam style LoRA")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--url")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--dry-run", action="store_true", help="print the ComfyUI API graph and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if args.width % 16 or args.height % 16:
        die("width and height must be multiples of 16")

    # Build first with optimistic settings; model checks below may remove LoRAs
    # and switch to normal (non-Lightning) sampler settings.
    requested_lightning = bool(args.speed_lora) and not args.no_lightning
    requested_style = None if args.no_style_lora else (args.style_lora or None)
    steps = args.steps if args.steps is not None else (8 if requested_lightning else 24)
    cfg = args.cfg if args.cfg is not None else (1.0 if requested_lightning else 4.0)

    if args.dry_run:
        graph = build_graph(
            prompt=args.prompt,
            negative=args.negative,
            width=args.width,
            height=args.height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler=args.sampler,
            scheduler=args.scheduler,
            denoise=args.denoise,
            model=args.model,
            clip=args.clip,
            vae=args.vae,
            speed_lora=args.speed_lora if requested_lightning else None,
            style_lora=requested_style,
            style_strength=args.style_strength,
        )
        print(json.dumps({"prompt": graph}, indent=2, ensure_ascii=False))
        return 0

    settings = comfy_settings(args)
    describe_server(settings)

    unet_info = node_info(settings, "UnetLoaderGGUF")
    clip_info = node_info(settings, "CLIPLoaderGGUF")
    unet_options = combo_options(unet_info, "unet_name")
    clip_options = combo_options(clip_info, "clip_name")
    vae_options = combo_options(node_info(settings, "VAELoader"), "vae_name")
    lora_options = combo_options(node_info(settings, "LoraLoaderModelOnly"), "lora_name")

    ensure_available("diffusion model", args.model, unet_options)
    ensure_available("text encoder", args.clip, clip_options)
    ensure_available("VAE", args.vae, vae_options)
    clip_types = combo_options(clip_info, "type")
    if clip_types and "qwen_image" not in clip_types:
        die(
            "this ComfyUI build does not support CLIPLoaderGGUF type 'qwen_image'. "
            "Update ComfyUI in the image or use the official Runpod ComfyUI image."
        )

    speed_lora = args.speed_lora if requested_lightning else None
    if speed_lora and not ensure_available(
        "Lightning LoRA", speed_lora, lora_options, required=False
    ):
        speed_lora = None

    style_lora = requested_style
    if style_lora and not ensure_available(
        "Gundam style LoRA", style_lora, lora_options, required=False
    ):
        style_lora = None

    if not speed_lora:
        if args.steps is None:
            steps = 24
        if args.cfg is None:
            cfg = 4.0
        log("using normal sampler settings (about 20-30 steps, CFG 3-4.5)")

    graph = build_graph(
        prompt=args.prompt,
        negative=args.negative,
        width=args.width,
        height=args.height,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        denoise=args.denoise,
        model=args.model,
        clip=args.clip,
        vae=args.vae,
        speed_lora=speed_lora,
        style_lora=style_lora,
        style_strength=args.style_strength,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "last_prompt.json").write_text(json.dumps({"prompt": graph}, indent=2) + "\n")

    log(
        f"queuing {args.width}x{args.height}, seed={seed}, steps={steps}, cfg={cfg}, "
        f"style_lora={style_lora or 'off'}, lightning={bool(speed_lora)}"
    )
    prompt_id = queue_prompt(settings, graph)
    log(f"prompt id: {prompt_id}")

    entry = wait_for_result(settings, prompt_id, args.timeout)
    if not entry:
        die("generation failed; see the ComfyUI logs or the error above")

    images: list[dict[str, Any]] = []
    for node_output in (entry.get("outputs") or {}).values():
        images.extend(node_output.get("images") or [])
    if not images:
        die("ComfyUI finished but returned no images")

    for index, image in enumerate(images, start=1):
        destination = download_output(settings, image, output_dir, seed, index)
        print(destination)

    log(f"done in one prompt; seed={seed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
