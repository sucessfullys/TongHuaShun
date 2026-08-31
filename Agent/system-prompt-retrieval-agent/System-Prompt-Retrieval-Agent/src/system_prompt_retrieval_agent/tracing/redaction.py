from __future__ import annotations

import os
import re
from typing import Any, Iterable

_KEYS = ("OPENAI_API_KEY", "Google_API_KEY")


def redact_secrets(obj: Any, env_keys: Iterable[str] = _KEYS, *, redact_large_b64: bool = True) -> Any:
    """Recursively redact any string equal to a live env var value.
    Also truncates base64-looking strings longer than 1024 chars when redact_large_b64."""
    values = {os.environ.get(k, "") for k in env_keys}
    values.discard("")
    b64_re = re.compile(r"^[A-Za-z0-9+/=]{1000,}$")

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(walk(v) for v in x)
        if isinstance(x, str):
            s = x
            for v in values:
                if v and v in s:
                    s = s.replace(v, "***REDACTED***")
            if redact_large_b64 and len(s) > 1024 and b64_re.match(s):
                return f"<base64 redacted len={len(s)}>"
            return s
        return x

    return walk(obj)
