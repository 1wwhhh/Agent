from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

_ENV_LOCK = Lock()
_ENV_LOADED = False


def load_project_env(*, override: bool = False) -> None:
    global _ENV_LOADED

    if _ENV_LOADED and not override:
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        _ENV_LOADED = True
        return

    with _ENV_LOCK:
        if _ENV_LOADED and not override:
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            env_key = key.strip()
            env_value = value.strip().strip('"').strip("'")
            if not env_key:
                continue
            if override or env_key not in os.environ:
                os.environ[env_key] = env_value

        _ENV_LOADED = True
