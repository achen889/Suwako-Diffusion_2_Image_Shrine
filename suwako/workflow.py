"""One generation path shared by Streamlit and the command-line interface."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json

from .history import OUTPUT_ROOT, save_run
from .models import ImageRequest
from .local import LocalCudaProvider
from .provider import OpenAIProvider

_LOCAL = LocalCudaProvider()


def run_generation(request: ImageRequest, root: Path = OUTPUT_ROOT, provider=None, *,
                   strategy="batch", progress_callback=None, image_callback=None,
                   stage_callback=None, prompt_sources: dict | None = None,
                   input_asset_id: str | None = None) -> Path:
    request.validate()
    backend = provider or (_LOCAL if request.provider == "local" else OpenAIProvider())
    started = perf_counter()
    if stage_callback:
        stage_callback("Loading selected model or connecting to provider…")
    if request.provider == "local" and backend is _LOCAL:
        images, details = backend.generate(
            request, strategy=strategy, progress_callback=progress_callback,
            image_callback=image_callback,
        )
    else:
        images, details = backend.generate(request)
        if image_callback:
            for index, image in enumerate(images, start=1):
                image_callback(image, index, len(images))
        if progress_callback:
            progress_callback({"fraction": 1.0, "completed_images": len(images),
                               "count": len(images), "strategy": "api"})
    generated = perf_counter()
    details["inference_seconds"] = round(generated - started, 3)
    if prompt_sources is not None:
        details["prompt_sources"] = prompt_sources
    if input_asset_id is not None:
        details["input_asset_id"] = input_asset_id
    if stage_callback:
        stage_callback("Saving generated images and metadata…")
    directory = save_run(request, images, details, generated - started, root)
    details["save_seconds"] = round(perf_counter() - generated, 3)
    # Persist save time too, so benchmark runs distinguish inference from file writing.
    metadata = directory / "metadata.json"
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["save_seconds"] = details["save_seconds"]
    data["elapsed_seconds"] = round(details["inference_seconds"] + details["save_seconds"], 3)
    temp = metadata.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(metadata)
    if stage_callback:
        stage_callback("Generation and saving complete")
    return directory


def local_provider() -> LocalCudaProvider:
    return _LOCAL
