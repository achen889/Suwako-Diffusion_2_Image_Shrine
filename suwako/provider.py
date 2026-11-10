"""Cloud provider. OpenAI is deliberately optional, never the CUDA fallback."""
from __future__ import annotations

import base64
import os
from contextlib import ExitStack
from io import BytesIO
from PIL import Image

from .models import ImageRequest
from .prompts import compose


class OpenAIProvider:
    def generate(self, request: ImageRequest):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI generation requires OPENAI_API_KEY; local CUDA does not.")
        prompt, negative = compose(request)
        client = OpenAI(timeout=180, max_retries=1)
        options = dict(model=request.model, prompt=prompt,
                       size=f"{request.width}x{request.height}",
                       quality=request.quality, output_format="png", n=request.count)
        paths = tuple(([request.input_image] if request.input_image else []) +
                      list(request.reference_images))
        if paths:
            with ExitStack() as stack:
                handles = [stack.enter_context(p.open("rb")) for p in paths]
                response = client.images.edit(image=handles, **options)
        else:
            response = client.images.generate(**options)
        images = []
        for item in response.data:
            if not item.b64_json:
                raise RuntimeError("OpenAI returned an image without base64 payload.")
            with Image.open(BytesIO(base64.b64decode(item.b64_json))) as img:
                images.append(img.convert("RGB"))
        if not images:
            raise RuntimeError("OpenAI returned no images.")
        return images, {"prompt_sent": prompt, "negative_prompt_sent": negative,
                        "seed": None, "per_image_seeds": None, "device": "API"}
