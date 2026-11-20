#!/usr/bin/env python3
"""Download the Qwen-Image model stack into /workspace/models.

This runs inside the Runpod container before ComfyUI starts. It deliberately
uses only the Python standard library so it cannot break the pinned ComfyUI
Python environment.

Files that already exist and look complete are skipped. Interrupted downloads
resume from a .part file when the server supports HTTP Range requests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

DEFAULT_MODELS_DIR = "/workspace/models"
DEFAULT_CONFIG = "/opt/gundam/models.json"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "gundam-qwen-runpod/1.0"


def log(message: str) -> None:
    print(f"[models] {time.strftime('%H:%M:%S')} {message}", flush=True)


def human_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def headers_for(url: str) -> dict[str, str]:
    host = (urlparse(url).hostname or "").lower()
    headers = {"User-Agent": USER_AGENT}
    hf_token = env("HF_TOKEN")
    civitai_token = env("CIVITAI_TOKEN")
    if (host == "huggingface.co" or host.endswith(".huggingface.co")) and hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    if "civitai.com" in host and civitai_token:
        headers["Authorization"] = f"Bearer {civitai_token}"
    return headers


def model_url(entry: dict, hf_base_url: str) -> str:
    explicit = env(entry.get("url_env", "")) if entry.get("url_env") else ""
    if explicit:
        return explicit

    repo = entry.get("repo")
    filename = entry.get("file")
    if not repo or not filename:
        return ""

    base = env("HF_ENDPOINT", hf_base_url).rstrip("/")
    return f"{base}/{repo}/resolve/main/{quote(filename, safe='/')}"


def looks_like_error_page(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(1024).lstrip().lower()
    except OSError:
        return False
    return (
        head.startswith(b"<!doctype")
        or head.startswith(b"<html")
        or head.startswith(b"{\"error\"")
        or head.startswith(b"{\"errors\"")
    )


def download_once(url: str, part: Path, target: Path, headers: dict[str, str]) -> bool:
    """One attempt. Returns True only when the complete file is in place."""
    start = part.stat().st_size if part.exists() else 0
    request_headers = dict(headers)
    if start:
        request_headers["Range"] = f"bytes={start}-"

    request = Request(url, headers=request_headers)
    try:
        response = urlopen(request, timeout=120)
    except HTTPError as exc:
        if exc.code == 416 and start:
            content_range = exc.headers.get("Content-Range", "") or ""
            match = re.search(r"/(\d+)\s*$", content_range)
            if match and int(match.group(1)) == start:
                os.replace(part, target)
                log(f"already complete: {target} ({human_bytes(start)})")
                return True
            log("server rejected the resume offset; restarting from zero")
            part.unlink(missing_ok=True)
            return False
        raise

    with response:
        status = response.status
        if start and status != 206:
            log("server ignored the Range request; restarting from zero")
            part.unlink(missing_ok=True)
            return False

        total: int | None = None
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/(\d+)\s*$", content_range)
            if match:
                total = int(match.group(1))
        else:
            content_length = response.headers.get("Content-Length", "")
            if content_length.isdigit():
                total = int(content_length)

        mode = "ab" if start else "wb"
        downloaded = start
        last_report = 0.0
        last_pct = -1
        part.parent.mkdir(parents=True, exist_ok=True)

        log(
            f"downloading {target.name} "
            f"({human_bytes(start)} already present"
            + (f", {human_bytes(total)} total)" if total else ")")
        )
        with part.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                pct = int(downloaded * 100 / total) if total else None
                report = now - last_report >= 5
                if pct is not None and pct >= last_pct + 5:
                    report = True
                if report:
                    if total:
                        log(
                            f"  {target.name}: {pct}% "
                            f"({human_bytes(downloaded)} / {human_bytes(total)})"
                        )
                    else:
                        log(f"  {target.name}: {human_bytes(downloaded)}")
                    last_report = now
                    if pct is not None:
                        last_pct = pct

    actual = part.stat().st_size if part.exists() else 0
    if total is not None and actual != total:
        log(
            f"incomplete download for {target.name}: "
            f"{actual} of {total} bytes; keeping .part for resume"
        )
        return False

    if looks_like_error_page(part):
        log(
            f"{target.name} looks like an HTML/JSON error page. "
            "Check the URL and CIVITAI_TOKEN/HF_TOKEN."
        )
        part.unlink(missing_ok=True)
        return False

    os.replace(part, target)
    log(f"ready: {target} ({human_bytes(actual)})")
    return True


def download_with_aria2(
    url: str,
    target: Path,
    min_bytes: int,
    headers: dict[str, str],
    expected_size: int | None = None,
) -> bool:
    """Use aria2c's multi-connection downloader when it is available."""
    aria2c = shutil.which("aria2c")
    if not aria2c:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (looks_like_error_page(target) or target.stat().st_size == 0):
        target.unlink()
    if target.exists() and expected_size and target.stat().st_size == expected_size:
        log(f"skip (size matches remote): {target}")
        return True
    if target.exists() and expected_size and target.stat().st_size > expected_size:
        log(f"removing oversized file: {target}")
        target.unlink()

    # Do not skip a file just because it is larger than min_bytes: an
    # interrupted 12 GB download is still smaller than the 13 GB original.
    # aria2c --continue checks the remote size and resumes or completes it.
    log(f"downloading with aria2c (16 connections): {target.name}")
    command = [
        aria2c,
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--check-certificate=true",
        "--max-tries=5",
        "--retry-wait=5",
        "--timeout=60",
        "--connect-timeout=30",
        "--summary-interval=15",
        "--console-log-level=notice",
        "--dir",
        str(target.parent),
        "--out",
        target.name,
        "--user-agent",
        USER_AGENT,
    ]
    authorization = headers.get("Authorization")
    if authorization:
        command.extend(["--header", f"Authorization: {authorization}"])
    command.append(url)

    try:
        completed = subprocess.run(command, timeout=6 * 3600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"aria2c failed to run: {exc}")
        return False

    if completed.returncode != 0:
        log(f"aria2c exited with status {completed.returncode}")
        return False
    if not target.exists():
        log("aria2c finished but the target file is missing")
        return False
    size = target.stat().st_size
    if looks_like_error_page(target):
        log("aria2c saved an HTML/JSON error page; deleting it")
        target.unlink(missing_ok=True)
        return False
    if expected_size and size != expected_size:
        log(
            f"aria2c size mismatch for {target.name}: "
            f"{human_bytes(size)} != {human_bytes(expected_size)}"
        )
        return False
    if size < min_bytes:
        log(f"aria2c produced an undersized file ({human_bytes(size)})")
        return False
    log(f"ready: {target} ({human_bytes(size)})")
    return True


def download_with_retries(
    url: str,
    target: Path,
    min_bytes: int,
    headers: dict[str, str],
    attempts: int = 5,
    expected_size: int | None = None,
) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    aria2_control = target.with_name(target.name + ".aria2")

    if target.exists() and looks_like_error_page(target):
        log(f"removing error page saved as {target}")
        target.unlink()

    if target.exists():
        size = target.stat().st_size
        if expected_size and size == expected_size:
            log(f"skip (size matches remote): {target}")
            part.unlink(missing_ok=True)
            aria2_control.unlink(missing_ok=True)
            return True
        if expected_size and size > expected_size:
            log(f"removing oversized file: {target}")
            target.unlink()
        elif not expected_size and size >= min_bytes and not aria2_control.exists() and not part.exists():
            log(f"skip (already present): {target} ({human_bytes(size)})")
            return True

    if download_with_aria2(url, target, min_bytes, headers, expected_size):
        part.unlink(missing_ok=True)
        aria2_control.unlink(missing_ok=True)
        return True

    # aria2c may have left a partial target behind. Move it to the .part name
    # so the standard-library fallback can resume it with an HTTP Range. If
    # both exist, keep whichever is larger.
    if target.exists():
        if looks_like_error_page(target):
            target.unlink(missing_ok=True)
        elif part.exists() and part.stat().st_size > target.stat().st_size:
            target.unlink()
        else:
            part.unlink(missing_ok=True)
            target.replace(part)

    for attempt in range(1, attempts + 1):
        try:
            if download_once(url, part, target, headers):
                size = target.stat().st_size
                if expected_size and size != expected_size:
                    log(
                        f"{target.name} size mismatch: "
                        f"{human_bytes(size)} != {human_bytes(expected_size)}"
                    )
                    target.unlink(missing_ok=True)
                    return False
                if size < min_bytes:
                    log(
                        f"{target.name} is smaller than expected "
                        f"({human_bytes(size)} < {human_bytes(min_bytes)})"
                    )
                    target.unlink(missing_ok=True)
                    return False
                aria2_control.unlink(missing_ok=True)
                return True
        except HTTPError as exc:
            if exc.code in (401, 403):
                log(
                    f"HTTP {exc.code} for {url}. "
                    "This usually means the download needs a valid token "
                    "(CIVITAI_TOKEN for Civitai, HF_TOKEN for gated HF repos)."
                )
                return False
            if exc.code == 404:
                log(f"HTTP 404 for {url}. Check the model URL/file name.")
                return False
            log(f"HTTP {exc.code} on attempt {attempt}/{attempts}: {exc.reason}")
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            log(f"network error on attempt {attempt}/{attempts}: {exc}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the pod booting
            log(f"unexpected error on attempt {attempt}/{attempts}: {exc!r}")

        if attempt < attempts:
            delay = min(60, 2**attempt)
            log(f"retrying in {delay}s...")
            time.sleep(delay)

    log(f"FAILED after {attempts} attempts: {target.name}")
    return False


def main() -> int:
    models_dir = Path(env("MODELS_DIR", DEFAULT_MODELS_DIR))
    config_path = Path(env("MODELS_CONFIG", DEFAULT_CONFIG))

    if env("SKIP_MODEL_DOWNLOAD") == "1":
        log("SKIP_MODEL_DOWNLOAD=1; not downloading anything")
        return 0

    if not config_path.exists():
        log(f"model config not found: {config_path}")
        return 1

    config = json.loads(config_path.read_text())
    hf_base_url = config.get("hf_base_url", "https://huggingface.co")
    models_dir.mkdir(parents=True, exist_ok=True)

    for subdir in (
        "diffusion_models",
        "text_encoders",
        "vae",
        "loras",
        "unet",
        "checkpoints",
        "controlnet",
        "clip_vision",
        "style_models",
        "embeddings",
        "upscale_models",
    ):
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(models_dir).free
    log(f"model directory: {models_dir} ({human_bytes(free)} free)")
    if free < 30 * 1024**3:
        log(
            "WARNING: less than 30 GiB free. The default Qwen-Image stack needs "
            "~20 GB plus outputs and the ComfyUI workspace."
        )

    failed_required: list[str] = []
    installed: list[dict[str, object]] = []

    for entry in config.get("required", []):
        url = model_url(entry, hf_base_url)
        if not url:
            log(f"missing URL/repo for required model {entry.get('name')}")
            failed_required.append(str(entry.get("name")))
            continue
        target = models_dir / entry["dest"] / entry["name"]
        ok = download_with_retries(
            url,
            target,
            int(entry.get("min_bytes", 1)),
            headers_for(url),
            expected_size=int(entry["size"]) if entry.get("size") else None,
        )
        if ok:
            installed.append(
                {"name": entry["name"], "path": str(target), "size": target.stat().st_size}
            )
        else:
            failed_required.append(str(entry.get("name")))

    for entry in config.get("optional", []):
        url_env = entry.get("url_env", "")
        url = model_url(entry, hf_base_url) if url_env else ""
        if not url:
            if url_env == "GUNDAM_LORA_URL":
                log(
                    "no GUNDAM_LORA_URL set; skipping the style LoRA. "
                    "The base Qwen-Image workflows still work."
                )
            continue

        filename = env(entry.get("filename_env", ""), str(entry.get("name", "")))
        target = models_dir / entry["dest"] / filename
        ok = download_with_retries(
            url,
            target,
            int(entry.get("min_bytes", 1)),
            headers_for(url),
        )
        if ok:
            installed.append({"name": filename, "path": str(target), "size": target.stat().st_size})
        else:
            log(
                "optional style LoRA failed to download. "
                "ComfyUI will still start; add the LoRA manually later."
            )

    manifest = models_dir / ".installed.json"
    manifest.write_text(json.dumps({"installed": installed}, indent=2) + "\n")

    if failed_required:
        log("required models are missing: " + ", ".join(failed_required))
        log(
            "ComfyUI will still start so you can inspect the volume/network, "
            "but workflows that need these files will fail."
        )
        return 1

    log("all required models are present")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
