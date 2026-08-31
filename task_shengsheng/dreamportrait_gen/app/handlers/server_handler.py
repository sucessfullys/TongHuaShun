#!/usr/bin/env python3

import base64
import logging
import time
from io import BytesIO

import tornado.escape
from PIL import Image
from tornado.options import options

from app.handlers.base_handler import BaseApiHandler
from app.modules.request_logger import RequestLogger, create_request_id


class DreamFaceHandler(BaseApiHandler):
    def initialize(self, model, *args, **kwargs):
        self.logger = logging.getLogger("handler")
        self.model = model
        self.request_logger = RequestLogger(
            enabled=options.enable_request_log,
            log_dir=options.request_log_dir,
            save_request_images=options.save_request_images,
            save_result_images=options.save_result_images,
        )
        super().initialize(*args, **kwargs)

    def _parse_body(self):
        content_type = self.request.headers.get("Content-Type", "")
        if "application/json" in content_type:
            body = self.request.body or b"{}"
            data = tornado.escape.json_decode(body)
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object")
            return data

        data = {
            "prompt": self.get_argument("prompt", default=""),
            "seed": self.get_argument("seed", default=str(42)),
            "steps": self.get_argument("steps", default=str(options.default_steps)),
            "cfg": self.get_argument("cfg", default=str(options.default_cfg)),
            "height": self.get_argument("height", default=str(options.default_height)),
            "width": self.get_argument("width", default=str(options.default_width)),
        }
        pics = self.get_arguments("pics")
        if pics:
            data["pics"] = pics
        else:
            pic = self.get_argument("pic", default="")
            if pic:
                data["pic"] = pic
        return data

    def _get_reference_payloads(self, data):
        pics = data.get("pics")
        if pics is None:
            pic = data.get("pic")
            pics = [pic] if pic else []
        elif isinstance(pics, str):
            pics = [pics]
        elif not isinstance(pics, list):
            raise ValueError("pics must be a list of base64 images")

        max_reference_images = int(options.max_reference_images)
        if len(pics) > max_reference_images:
            raise ValueError(f"Too many reference images: {len(pics)} > {max_reference_images}")
        return pics

    def _decode_image(self, b64_string):
        if not isinstance(b64_string, str) or not b64_string.strip():
            raise ValueError("Reference image must be a non-empty base64 string")

        b64_string = b64_string.strip()
        if "," in b64_string and b64_string.split(",", 1)[0].startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]

        raw = base64.b64decode(b64_string)
        image = Image.open(BytesIO(raw)).convert("RGB")
        width, height = image.size
        if width * height > int(options.max_image_pixels):
            raise ValueError(
                f"Reference image too large: {width}x{height}, max pixels={options.max_image_pixels}"
            )
        return image

    def _encode_image(self, image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _optional_int(self, data, key):
        value = data.get(key)
        if value is None or value == "":
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        return value

    def _optional_float(self, data, key):
        value = data.get(key)
        if value is None or value == "":
            return None
        return float(value)

    def _build_log_record(
        self,
        *,
        request_id,
        started_at,
        prompt="",
        params=None,
        images=None,
        input_image_paths=None,
        output_image_path=None,
        status="error",
        error=None,
    ):
        return {
            "request_id": request_id,
            "client_ip": self.request.remote_ip,
            "prompt": prompt,
            "params": params or {},
            "input_image_count": len(images or []),
            "input_image_paths": input_image_paths or [],
            "output_image_path": output_image_path,
            "elapsed_ms": int((time.time() - started_at) * 1000),
            "status": status,
            "error": error,
        }

    async def post(self):
        request_id = create_request_id()
        started_at = time.time()
        prompt = ""
        params = {}
        images = []
        input_image_paths = []
        output_image_path = None

        try:
            data = self._parse_body()
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Missing argument: prompt")

            reference_payloads = self._get_reference_payloads(data)
            images = [self._decode_image(item) for item in reference_payloads]
            input_image_paths = self.request_logger.save_input_images(request_id, images)

            params = {
                "seed": int(data.get("seed", 42)),
                "steps": self._optional_int(data, "steps"),
                "cfg": self._optional_float(data, "cfg"),
                "height": self._optional_int(data, "height"),
                "width": self._optional_int(data, "width"),
            }

            result = self.model.infer(
                prompt,
                images,
                seed=params["seed"],
                steps=params["steps"],
                cfg=params["cfg"],
                height=params["height"],
                width=params["width"],
            )
            output_image_path = self.request_logger.save_output_image(request_id, result)
            b64_result = self._encode_image(result)
        except Exception as e:
            logging.error("DreamFace inference error: %s", e)
            self.request_logger.write(
                self._build_log_record(
                    request_id=request_id,
                    started_at=started_at,
                    prompt=prompt,
                    params=params,
                    images=images,
                    input_image_paths=input_image_paths,
                    output_image_path=output_image_path,
                    status="error",
                    error=str(e),
                )
            )
            self.respond({}, 1, str(e))
        else:
            self.request_logger.write(
                self._build_log_record(
                    request_id=request_id,
                    started_at=started_at,
                    prompt=prompt,
                    params=params,
                    images=images,
                    input_image_paths=input_image_paths,
                    output_image_path=output_image_path,
                    status="success",
                )
            )
            self.respond(b64_result, 0)
