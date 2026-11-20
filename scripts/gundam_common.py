"""Small shared helpers for the local launcher and generator scripts."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
STATE_FILE = ROOT / ".runpod-state.json"
API_BASE = "https://api.runpod.io/v2"

_ENV_CACHE: dict[str, str] | None = None


class HttpError(RuntimeError):
    def __init__(self, status: int, detail: str, body: Any = None):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail
        self.body = body


def load_dotenv(path: Path = ENV_FILE) -> dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE

    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            elif " #" in value:
                # Allow inline comments on unquoted values.
                value = value.split(" #", 1)[0].rstrip()
            values[key] = value
    _ENV_CACHE = values
    return values


def setting(key: str, default: str = "") -> str:
    """Environment variables win over .env, which wins over the default."""
    value = os.environ.get(key)
    if value is not None and value != "":
        return value
    return load_dotenv().get(key, default)


def setting_int(key: str, default: int) -> int:
    raw = setting(key, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def setting_bool(key: str, default: bool = False) -> bool:
    raw = setting(key, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    try:
        STATE_FILE.chmod(0o600)
    except OSError:
        pass


def clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def basic_auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> Any:
    request_headers = {"Accept": "application/json", "User-Agent": "gundam-qwen-runpod/1.0"}
    if headers:
        request_headers.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(url, data=data, method=method, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if not payload:
                    return None
                return json.loads(payload)
        except HTTPError as exc:
            payload = exc.read()
            parsed: Any = None
            if payload:
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = payload.decode(errors="replace")
            detail = ""
            if isinstance(parsed, dict):
                detail = str(parsed.get("detail") or parsed.get("message") or parsed)
                if parsed.get("errors"):
                    detail += f" ({parsed['errors']})"
            else:
                detail = str(parsed or exc.reason)

            if exc.code == 429 and attempt < retries:
                wait = int(exc.headers.get("Retry-After", "5") or 5)
                time.sleep(max(1, wait))
                last_error = HttpError(exc.code, detail, parsed)
                continue
            if 500 <= exc.code < 600 and attempt < retries:
                time.sleep(2**attempt)
                last_error = HttpError(exc.code, detail, parsed)
                continue
            raise HttpError(exc.code, detail or str(exc.reason), parsed) from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
                continue

    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def api_request(
    method: str,
    path: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> Any:
    if not path.startswith("/"):
        path = "/" + path
    return request_json(
        method,
        API_BASE + path,
        body=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        retries=retries,
    )


def ui_url(pod_id: str, port: int = 8188) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"
