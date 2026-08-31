from __future__ import annotations

import logging
import os
from typing import Iterable


class RedactionFilter(logging.Filter):
    """Strip any environment-variable value found in the log message or args."""

    def __init__(self, env_keys: Iterable[str]) -> None:
        super().__init__()
        self.env_keys = list(env_keys)

    def _values(self) -> list[str]:
        out = []
        for k in self.env_keys:
            v = os.environ.get(k)
            if v:
                out.append(v)
        return out

    def filter(self, record: logging.LogRecord) -> bool:
        vals = self._values()
        if not vals:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        replaced = msg
        for v in vals:
            if v and v in replaced:
                replaced = replaced.replace(v, "***REDACTED***")
        if replaced != msg:
            record.msg = replaced
            record.args = ()
        return True


def setup_logging(level: str = "INFO", env_keys: Iterable[str] = ("OPENAI_API_KEY", "Google_API_KEY")) -> logging.Logger:
    logger = logging.getLogger("system_prompt_retrieval_agent")
    logger.setLevel(getattr(logging, level, logging.INFO))
    if not any(isinstance(f, RedactionFilter) for f in logger.filters):
        logger.addFilter(RedactionFilter(env_keys))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
        h.addFilter(RedactionFilter(env_keys))
        logger.addHandler(h)
    return logger
