"""Persist Gradio PiD inference runs: params, timings, prompt, input/output images."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

_SHENSHENG_ROOT = os.environ.get("SHENSHENG_ROOT", "/mnt/data/image-edit/datasets/shensheng")
DEFAULT_LOG_DIR = Path(_SHENSHENG_ROOT) / "outputs" / "gradio_pid_logs"
IMAGE_FORMAT = "JPEG"
IMAGE_QUALITY = 92


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _args_to_dict(args_ns: SimpleNamespace) -> dict[str, Any]:
    fields = (
        "pid_root",
        "backbone",
        "backbone_model_id",
        "pid_ckpt_type",
        "checkpoint_path",
        "experiment",
        "resolution",
        "scale",
        "ldm_steps",
        "guidance_scale",
        "pid_steps",
        "pid_cfg",
        "seed",
        "lora",
        "lora_scale",
        "low_vram",
        "local_files_only",
        "no_lora",
    )
    out: dict[str, Any] = {}
    for key in fields:
        if hasattr(args_ns, key):
            value = getattr(args_ns, key)
            if value is not None:
                out[key] = value
    return out


def _timings_to_dict(timings: list[tuple[str, float]], total_sec: float) -> dict[str, Any]:
    steps = {name: round(sec, 3) for name, sec in timings}
    return {"steps": steps, "total_sec": round(total_sec, 3)}


def _save_image(img: Image.Image, path: Path) -> None:
    img.convert("RGB").save(
        path,
        format=IMAGE_FORMAT,
        quality=IMAGE_QUALITY,
        subsampling=2,
        optimize=False,
    )


class GradioInferLogger:
    def __init__(self, log_dir: Path | str, *, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self._lock = threading.Lock()

    def _new_run_dir(self) -> Path:
        now = _utc_now()
        run_id = f"{now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = self.log_dir / now.strftime("%Y%m%d") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        jsonl_path = self.log_dir / "runs.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_run(
        self,
        *,
        prompt: str,
        args_ns: SimpleNamespace,
        input_image: Image.Image,
        preprocessed_image: Image.Image | None,
        output_image: Image.Image | None,
        prep_info: str,
        resolution_info: str,
        timings: list[tuple[str, float]],
        total_sec: float,
        status: str = "ok",
        error: str = "",
    ) -> Path | None:
        if not self.enabled:
            return None

        with self._lock:
            run_dir = self._new_run_dir()
            ext = "jpg"

            input_path = run_dir / f"input.{ext}"
            _save_image(input_image, input_path)

            preprocessed_path = ""
            if preprocessed_image is not None:
                preprocessed_path = str(run_dir / f"preprocessed.{ext}")
                _save_image(preprocessed_image, Path(preprocessed_path))

            output_path = ""
            if output_image is not None:
                output_path = str(run_dir / f"output.{ext}")
                _save_image(output_image, Path(output_path))

            prompt_path = run_dir / "prompt.txt"
            prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")

            meta = {
                "timestamp_utc": _utc_now().isoformat(),
                "status": status,
                "error": error,
                "prompt": prompt.strip(),
                "params": _args_to_dict(args_ns),
                "prep_info": prep_info,
                "resolution_info": resolution_info,
                "timings": _timings_to_dict(timings, total_sec),
                "files": {
                    "input": str(input_path),
                    "preprocessed": preprocessed_path or None,
                    "output": output_path or None,
                    "prompt": str(prompt_path),
                },
                "run_dir": str(run_dir),
            }
            self._write_json(run_dir / "meta.json", meta)
            self._append_jsonl(
                {
                    "timestamp_utc": meta["timestamp_utc"],
                    "status": status,
                    "prompt": meta["prompt"],
                    "params": meta["params"],
                    "timings": meta["timings"],
                    "run_dir": str(run_dir),
                }
            )

            print(f"[log] saved run -> {run_dir}", flush=True)
            return run_dir

    def submit_run(self, **kwargs) -> None:
        """Write log in background so inference response is not blocked."""
        if not self.enabled:
            return

        def _target() -> None:
            time.sleep(0.5)
            self.log_run(**kwargs)

        threading.Thread(target=_target, daemon=True).start()
