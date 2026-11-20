#!/usr/bin/env python3
"""Create and manage a Runpod RTX 3090 Pod for Gundam Qwen-Image.

Examples:
    cp .env.example .env && $EDITOR .env
    python3 scripts/launch.py up
    python3 scripts/launch.py logs
    python3 scripts/launch.py down --yes
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from gundam_common import (
    HttpError,
    api_request,
    clear_state,
    format_duration,
    read_state,
    setting,
    setting_bool,
    setting_int,
    ui_url,
    write_state,
)

PLACEHOLDER_PASSWORDS = {
    "",
    "change-me",
    "changeme",
    "password",
    "replace-with-a-long-random-password",
}
PLACEHOLDER_IMAGES = {
    "",
    "ghcr.io/your-github-user/gundam-qwen:latest",
    "ghcr.io/YOUR_GITHUB_USER/gundam-qwen:latest",
}


def log(message: str) -> None:
    print(f"[launch] {message}", flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"[launch] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def load_settings() -> dict[str, Any]:
    return {
        "api_key": setting("RUNPOD_API_KEY"),
        "image": setting("RUNPOD_IMAGE"),
        "bootstrap_url": setting("RUNPOD_BOOTSTRAP_URL"),
        "gpu": setting("RUNPOD_GPU", "NVIDIA GeForce RTX 3090"),
        "cloud": setting("RUNPOD_CLOUD", "COMMUNITY").upper(),
        "region": setting("RUNPOD_REGION").upper(),
        "data_center": setting("RUNPOD_DATA_CENTER_ID").upper(),
        "network_volume": setting("RUNPOD_NETWORK_VOLUME_ID"),
        "container_disk": setting_int("RUNPOD_CONTAINER_DISK_GB", 40),
        "volume": setting_int("RUNPOD_VOLUME_GB", 50),
        "pod_name": setting("RUNPOD_POD_NAME", "gundam-qwen"),
        "auth_user": setting("BASIC_AUTH_USER", "gundam"),
        "auth_password": setting("BASIC_AUTH_PASSWORD"),
        "lora_url": setting("GUNDAM_LORA_URL"),
        "lora_filename": setting("GUNDAM_LORA_FILENAME", "gundam_qwen_image.safetensors"),
        "civitai_token": setting("CIVITAI_TOKEN"),
        "hf_token": setting("HF_TOKEN"),
        "registry_id": setting("RUNPOD_REGISTRY_AUTH_ID"),
        "cache_none": setting_bool("COMFYUI_CACHE_NONE", False),
        "extra_args": setting("COMFYUI_EXTRA_ARGS"),
        "http_port": setting_int("RUNPOD_HTTP_PORT", 8188),
    }


def validate_settings(s: dict[str, Any]) -> None:
    if not s["api_key"]:
        die(
            "RUNPOD_API_KEY is not set. Create a key at "
            "https://console.runpod.io/user/credentials and put it in .env."
        )
    if not s["bootstrap_url"] and (s["image"] in PLACEHOLDER_IMAGES or not s["image"]):
        die(
            "RUNPOD_IMAGE is not set. Use the GHCR image built by GitHub Actions, "
            "for example ghcr.io/<your-github-user>/<repo>:latest, or set "
            "RUNPOD_BOOTSTRAP_URL for the public-base bootstrap path."
        )
    if s["cloud"] not in {"SECURE", "COMMUNITY"}:
        die("RUNPOD_CLOUD must be SECURE or COMMUNITY.")
    if s["auth_password"] in PLACEHOLDER_PASSWORDS:
        die(
            "BASIC_AUTH_PASSWORD is unset or still the example value. "
            "Set a long random password in .env; this is the password for the UI."
        )
    if ":" in s["auth_user"]:
        die("BASIC_AUTH_USER must not contain a colon.")
    if s["container_disk"] < 20:
        die("RUNPOD_CONTAINER_DISK_GB must be at least 20.")
    if not s["network_volume"] and s["volume"] < 30:
        die("RUNPOD_VOLUME_GB must be at least 30 for the ~20 GB model stack.")


def get_network_volume(volume_id: str, api_key: str) -> dict[str, Any]:
    try:
        return api_request("GET", f"/network-volumes/{volume_id}", api_key)
    except HttpError as exc:
        die(f"could not read network volume {volume_id}: {exc}")


def candidate_datacenters(s: dict[str, Any], volume: dict[str, Any] | None) -> list[str | None]:
    volume_dc: str | None = None
    if volume:
        volume_dc = volume.get("dataCenter")
        if not volume_dc:
            die("the network volume has no dataCenter field; cannot place the pod")

    if s["data_center"]:
        if volume_dc and volume_dc != s["data_center"]:
            die(
                f"RUNPOD_DATA_CENTER_ID={s['data_center']} does not match the "
                f"network volume's data center ({volume_dc})."
            )
        return [s["data_center"]]

    if volume_dc:
        return [str(volume_dc)]

    params: dict[str, str] = {"include": "GPU_AVAILABILITY"}
    if s["region"]:
        params["regions"] = s["region"]

    try:
        catalog = api_request(
            "GET",
            "/catalog/datacenters?" + urlencode(params),
            s["api_key"],
        )
    except (HttpError, RuntimeError) as exc:
        log(f"could not read the data-center catalog: {exc}; letting Runpod choose")
        return [None]

    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    candidates: list[tuple[int, str]] = []
    for data_center in catalog.get("dataCenters", []):
        for gpu in data_center.get("gpuAvailability") or []:
            if gpu.get("id") != s["gpu"]:
                continue
            availability = gpu.get("availability")
            if availability in rank:
                candidates.append((rank[availability], str(data_center.get("id"))))
                break

    candidates.sort()
    ordered = [data_center for _, data_center in candidates]
    if ordered:
        log(
            f"{s['gpu']} availability: "
            + ", ".join(ordered[:8])
            + (" ..." if len(ordered) > 8 else "")
        )
    else:
        log(
            f"catalog reports no {s['gpu']} stock"
            + (f" in {s['region']}" if s["region"] else "")
            + "; trying Runpod's scheduler anyway"
        )
        return [None]
    return ordered


def build_pod_body(
    s: dict[str, Any],
    data_center: str | None,
    volume: dict[str, Any] | None,
) -> dict[str, Any]:
    env = {
        "BASIC_AUTH_USER": s["auth_user"],
        "BASIC_AUTH_PASSWORD": s["auth_password"],
        "GUNDAM_LORA_FILENAME": s["lora_filename"],
        "COMFYUI_CACHE_NONE": "1" if s["cache_none"] else "0",
        "PUBLIC_PORT": str(s["http_port"]),
    }
    if s["lora_url"]:
        env["GUNDAM_LORA_URL"] = s["lora_url"]
    if s["civitai_token"]:
        env["CIVITAI_TOKEN"] = s["civitai_token"]
    if s["hf_token"]:
        env["HF_TOKEN"] = s["hf_token"]
    if s["extra_args"]:
        env["COMFYUI_EXTRA_ARGS"] = s["extra_args"]

    if volume:
        mounts: dict[str, Any] = {
            "network": [{"volumeId": s["network_volume"], "path": "/workspace"}]
        }
    else:
        mounts = {"persistent": {"size": s["volume"], "path": "/workspace"}}

    body: dict[str, Any] = {
        "name": s["pod_name"],
        "image": s["image"],
        "gpu": {
            "id": s["gpu"],
            "count": 1,
            "minRamPerGpu": 32,
            "minVcpuCountPerGpu": 8,
        },
        "cloud": s["cloud"],
        "disk": s["container_disk"],
        "ports": [f"{s['http_port']}/http", "22/tcp"],
        "env": env,
        "mounts": mounts,
        "startSsh": True,
        "startJupyter": False,
    }
    if data_center:
        body["dataCenterIds"] = [data_center]

    if s.get("bootstrap_url"):
        # Use the public ComfyUI image and bootstrap the GGUF/model setup from
        # the public GitHub repo. No GHCR registry credential is needed.
        body["image"] = "runpod/comfyui:latest"
        body["entrypoint"] = ["/bin/bash", "-lc"]
        body["cmd"] = [f"curl -fsSL {shlex.quote(s['bootstrap_url'])} | bash"]
    elif s["registry_id"]:
        body["registry"] = s["registry_id"]
    return body


def find_existing_pod(api_key: str, name: str) -> dict[str, Any] | None:
    try:
        result = api_request("GET", "/pods?limit=100", api_key)
    except (HttpError, RuntimeError):
        return None
    for pod in result.get("pods", []):
        if pod.get("name") == name and pod.get("status") != "TERMINATED":
            return pod
    return None


def create_pod(s: dict[str, Any]) -> dict[str, Any]:
    volume = None
    if s["network_volume"]:
        volume = get_network_volume(s["network_volume"], s["api_key"])
        size = volume.get("size")
        if isinstance(size, int) and size < 30:
            die(
                f"network volume {s['network_volume']} is only {size} GB; "
                "the default model stack needs at least 30 GB (50 GB recommended)."
            )
        log(
            f"using network volume {s['network_volume']} "
            f"({size} GB) in {volume.get('dataCenter')}"
        )

    candidates = candidate_datacenters(s, volume)
    last_detail = ""
    for data_center in candidates:
        body = build_pod_body(s, data_center, volume)
        label = data_center or "scheduler-chosen data center"
        log(f"requesting RTX 3090 in {label}...")
        try:
            pod = api_request("POST", "/pods", s["api_key"], body, retries=1)
        except HttpError as exc:
            last_detail = exc.detail
            if exc.status == 402:
                die(f"insufficient Runpod balance: {exc.detail}")
            if exc.status == 422:
                die(f"Runpod rejected the pod request: {exc.detail}")
            if exc.status in {400, 403, 409}:
                log(f"  not placed ({exc.status}): {exc.detail}")
                continue
            die(f"pod creation failed: {exc}")

        if not pod or not pod.get("id"):
            die(f"unexpected pod-create response: {pod!r}")

        log(f"created pod {pod['id']} in {pod.get('dataCenterId')}")
        return pod

    die(
        "no data center accepted the pod"
        + (f": {last_detail}" if last_detail else "")
        + ". If this is capacity, try RUNPOD_CLOUD=SECURE, RUNPOD_REGION=..., "
        "or RUNPOD_GPU='NVIDIA GeForce RTX 4090'. If it is a request error, "
        "check RUNPOD_IMAGE, RUNPOD_REGISTRY_AUTH_ID, and the volume settings."
    )


def health_ok(url: str, timeout: int = 10) -> bool:
    request = Request(f"{url}/healthz", headers={"User-Agent": "gundam-qwen-runpod/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, TimeoutError, ConnectionError, OSError):
        return False


def get_pod(api_key: str, pod_id: str) -> dict[str, Any]:
    return api_request("GET", f"/pods/{pod_id}", api_key)


def wait_for_ready(s: dict[str, Any], pod_id: str, timeout: int) -> dict[str, Any] | None:
    url = ui_url(pod_id, s["http_port"])
    started = time.monotonic()
    last_status = ""
    last_note = 0.0

    while time.monotonic() - started < timeout:
        try:
            pod = get_pod(s["api_key"], pod_id)
        except HttpError as exc:
            if exc.status == 404:
                log("pod no longer exists")
                return None
            log(f"status check failed: {exc}; retrying")
            time.sleep(10)
            continue

        status = str(pod.get("status", "UNKNOWN"))
        if status != last_status:
            log(f"pod status: {status}")
            last_status = status

        if status == "RUNNING" and health_ok(url):
            return pod

        if status in {"ERROR", "EXITED", "TERMINATED"}:
            log(f"pod entered {status}; it will not become ready on its own")
            return None

        now = time.monotonic()
        if now - last_note >= 30:
            elapsed = format_duration(now - started)
            if status == "RUNNING":
                log(
                    f"container is up; waiting for model download + ComfyUI "
                    f"({elapsed} elapsed)"
                )
            else:
                log(f"waiting for pod ({elapsed} elapsed, status {status})")
            last_note = now
        time.sleep(5)

    log(
        f"timed out after {format_duration(timeout)}. "
        f"Follow logs with: python3 scripts/launch.py logs --pod-id {pod_id}"
    )
    return None


def print_ready(s: dict[str, Any], pod: dict[str, Any]) -> None:
    pod_id = pod["id"]
    url = ui_url(pod_id, s["http_port"])
    ssh = pod.get("ssh") or {}
    direct = ssh.get("direct") or {}
    proxy = ssh.get("proxy") or {}
    ssh_command = direct.get("command") or proxy.get("command") or "(no SSH key registered)"

    state = read_state()
    state.update(
        {
            "pod_id": pod_id,
            "name": pod.get("name"),
            "url": url,
            "data_center": pod.get("dataCenterId"),
            "cloud": pod.get("cloud"),
            "gpu": (pod.get("gpu") or {}).get("id"),
            "cost_per_hour": pod.get("cost"),
            "ssh_command": ssh_command,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_state(state)

    print()
    print("=" * 72)
    print("  Gundam Qwen-Image is ready")
    print("=" * 72)
    print(f"  UI:       {url}")
    print(f"  User:     {s['auth_user']}")
    print(f"  Password: (BASIC_AUTH_PASSWORD from .env)")
    print(f"  Pod:      {pod_id}  ({pod.get('dataCenterId')}, {pod.get('cloud')})")
    if pod.get("cost"):
        print(f"  GPU:      {(pod.get('gpu') or {}).get('id')}  (${pod['cost']}/hr)")
    print(f"  SSH:      {ssh_command}")
    print()
    print("  Load workflow 'gundam_qwen_image' in the ComfyUI workflow browser,")
    print("  or 'qwen_image_base' if you have not added a style LoRA yet.")
    print()
    print("  Stop compute: python3 scripts/launch.py stop")
    print("  Terminate:    python3 scripts/launch.py down --yes")
    print("=" * 72)


def cmd_up(args: argparse.Namespace) -> int:
    s = load_settings()
    validate_settings(s)

    state = read_state()
    if state.get("pod_id"):
        try:
            existing = get_pod(s["api_key"], state["pod_id"])
            if existing.get("status") != "TERMINATED":
                die(
                    f"pod {state['pod_id']} already exists ({existing.get('status')}). "
                    "Use 'start' to resume it, 'status' to inspect it, or 'down --yes' "
                    "to delete it before creating a new one."
                )
        except HttpError:
            pass

    existing = find_existing_pod(s["api_key"], s["pod_name"])
    if existing:
        die(
            f"a pod named {s['pod_name']} already exists ({existing['id']}, "
            f"{existing.get('status')}). Use 'status --pod-id ...', 'start', or "
            "'down --yes' before creating a new one."
        )

    pod = create_pod(s)
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "pod_id": pod["id"],
        "name": pod.get("name"),
        "url": ui_url(pod["id"], s["http_port"]),
        "data_center": pod.get("dataCenterId"),
        "cloud": pod.get("cloud"),
        "gpu": (pod.get("gpu") or {}).get("id"),
        "cost_per_hour": pod.get("cost"),
        "created_at": now,
        "updated_at": now,
    }
    write_state(state)
    log(f"UI will be: {state['url']}")
    log("first boot downloads ~20 GB of models; this can take 10-20 minutes")

    if args.no_wait:
        log("--no-wait set; not waiting for ComfyUI")
        return 0

    ready = wait_for_ready(s, pod["id"], args.timeout)
    if not ready:
        log(f"pod {pod['id']} still exists. Inspect it with:")
        log(f"  python3 scripts/launch.py status --pod-id {pod['id']}")
        log(f"  python3 scripts/launch.py logs --pod-id {pod['id']}")
        log("stop compute with 'stop' or delete it with 'down --yes'")
        return 1

    print_ready(s, ready)
    return 0


def resolve_pod_id(s: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    state = read_state()
    if state.get("pod_id"):
        return str(state["pod_id"])
    die("no pod id. Pass --pod-id or run 'up' first.")


def confirm(prompt: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        die("refusing to continue without --yes in a non-interactive shell")
    answer = input(prompt).strip().lower()
    if answer not in {"y", "yes"}:
        die("cancelled")


def cmd_down(args: argparse.Namespace) -> int:
    s = load_settings()
    if not s["api_key"]:
        die("RUNPOD_API_KEY is not set")
    pod_id = resolve_pod_id(s, args.pod_id)
    if setting("RUNPOD_NETWORK_VOLUME_ID"):
        consequence = "The network volume itself is kept."
    else:
        consequence = "Its volume disk (including downloaded models) will be deleted."
    confirm(f"Terminate pod {pod_id}? {consequence} [y/N] ", args.yes)
    try:
        api_request("DELETE", f"/pods/{pod_id}", s["api_key"])
    except HttpError as exc:
        if exc.status != 404:
            die(f"could not terminate pod: {exc}")
    clear_state()
    log(f"terminated {pod_id}")
    return 0


def pod_action(args: argparse.Namespace, action: str, wait: bool = False) -> int:
    s = load_settings()
    if not s["api_key"]:
        die("RUNPOD_API_KEY is not set")
    pod_id = resolve_pod_id(s, args.pod_id)
    try:
        pod = api_request(
            "POST",
            f"/pods/{pod_id}/action",
            s["api_key"],
            {"action": action},
            retries=1,
        )
    except HttpError as exc:
        die(f"{action} failed: {exc}")
    log(f"{action} requested; status is {pod.get('status')}")
    if wait:
        ready = wait_for_ready(s, pod_id, args.timeout)
        if not ready:
            return 1
        print_ready(s, ready)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    s = load_settings()
    if not s["api_key"]:
        die("RUNPOD_API_KEY is not set")
    pod_id = resolve_pod_id(s, args.pod_id)
    try:
        pod = get_pod(s["api_key"], pod_id)
    except HttpError as exc:
        die(f"could not read pod {pod_id}: {exc}")

    print(f"Pod:       {pod.get('id')} ({pod.get('name')})")
    print(f"Status:    {pod.get('status')}")
    print(f"GPU:       {(pod.get('gpu') or {}).get('id')}")
    print(f"Cloud:     {pod.get('cloud')} / {pod.get('dataCenterId')}")
    cost = pod.get("cost")
    print(f"Cost:      ${cost}/hr (compute; storage is separate)" if cost is not None else "Cost:      unknown")
    print(f"UI:        {ui_url(pod_id, s['http_port'])}")
    print(f"Auth:      {s['auth_user']} / (BASIC_AUTH_PASSWORD from .env)")
    ssh = pod.get("ssh") or {}
    for label, endpoint in (("SSH direct", ssh.get("direct")), ("SSH proxy", ssh.get("proxy"))):
        if endpoint:
            print(f"{label:10} {endpoint.get('command')}")
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    s = load_settings()
    pod_id = resolve_pod_id(s, args.pod_id)
    print(ui_url(pod_id, s["http_port"]))
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    s = load_settings()
    if not s["api_key"]:
        die("RUNPOD_API_KEY is not set")
    pod_id = resolve_pod_id(s, args.pod_id)
    pod = get_pod(s["api_key"], pod_id)
    ssh = pod.get("ssh") or {}
    direct = ssh.get("direct") or {}
    proxy = ssh.get("proxy") or {}
    command = direct.get("command") or proxy.get("command")
    if not command:
        die(
            "no SSH command available. Add an SSH public key in Runpod credentials, "
            "and make sure the pod was created with startSsh."
        )
    print(command)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    s = load_settings()
    if not s["api_key"]:
        die("RUNPOD_API_KEY is not set")
    pod_id = resolve_pod_id(s, args.pod_id)
    url = (
        f"https://api.runpod.io/v2/pods/{pod_id}/logs?"
        + urlencode({"tail": args.tail})
    )
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {s['api_key']}",
            "Accept": "text/event-stream",
            "User-Agent": "gundam-qwen-runpod/1.0",
        },
    )
    log(f"streaming logs for {pod_id} (Ctrl+C to stop)")
    try:
        with urlopen(request, timeout=3600) as response:
            for raw in response:
                line = raw.decode(errors="replace").rstrip("\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    print(payload)
                    continue
                print(f"[{event.get('source', '?')}] {event.get('line', '')}", flush=True)
    except KeyboardInterrupt:
        return 0
    except HTTPError as exc:
        die(f"log stream failed: HTTP {exc.code}: {exc.read().decode(errors='replace')}")
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        die(f"log stream failed: {exc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="create a 3090 pod and wait for ComfyUI")
    up.add_argument("--no-wait", action="store_true", help="return right after creating the pod")
    up.add_argument("--timeout", type=int, default=2700, help="seconds to wait for readiness")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="terminate the pod (deletes its volume disk)")
    down.add_argument("--pod-id")
    down.add_argument("--yes", action="store_true")
    down.set_defaults(func=cmd_down)

    for name, help_text in (("stop", "stop the pod (keeps /workspace)"), ("start", "start a stopped pod")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--pod-id")
        command.add_argument("--timeout", type=int, default=2700)
        command.set_defaults(
            func=lambda args, action=name: pod_action(args, action, wait=(action == "start"))
        )

    restart = sub.add_parser("restart", help="restart the pod container")
    restart.add_argument("--pod-id")
    restart.add_argument("--timeout", type=int, default=2700)
    restart.set_defaults(func=lambda args: pod_action(args, "restart", wait=True))

    status = sub.add_parser("status", help="show pod status and URLs")
    status.add_argument("--pod-id")
    status.set_defaults(func=cmd_status)

    url = sub.add_parser("url", help="print the ComfyUI proxy URL")
    url.add_argument("--pod-id")
    url.set_defaults(func=cmd_url)

    ssh = sub.add_parser("ssh", help="print the SSH command")
    ssh.add_argument("--pod-id")
    ssh.set_defaults(func=cmd_ssh)

    logs = sub.add_parser("logs", help="stream pod logs")
    logs.add_argument("--pod-id")
    logs.add_argument("--tail", type=int, default=100, help="historical lines to replay (0-5000)")
    logs.set_defaults(func=cmd_logs)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
