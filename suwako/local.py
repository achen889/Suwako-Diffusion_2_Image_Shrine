"""One cached CUDA pipeline; batch inference with VRAM-aware retry."""
from __future__ import annotations

import gc
import secrets
import threading
from PIL import Image

from .models import LOCAL_MODELS, ImageRequest
from .prompts import compose


class LocalCudaProvider:
    def __init__(self):
        self.lock = threading.RLock()
        self.pipeline = None
        self.image_pipeline = None
        self.loaded_key = None

    @staticmethod
    def status() -> dict:
        try:
            import torch
        except ImportError:
            return {
                "cuda_available": False,
                "torch": None,
                "torch_cuda": None,
                "gpu": None,
                "vram_gb": None,
                "diagnostic": (
                    "PyTorch is not installed. Install a CUDA-enabled PyTorch build "
                    "for this Python environment, then run --status again."
                ),
            }
        cuda_available = torch.cuda.is_available()
        return {
            "cuda_available": cuda_available,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if cuda_available else None,
            "diagnostic": None if cuda_available else (
                "CUDA is unavailable. Install CUDA-enabled PyTorch and confirm the "
                "NVIDIA driver is visible; CPU fallback is disabled."
            ),
        }

    def unload(self):
        with self.lock:
            self.pipeline = None
            self.image_pipeline = None
            self.loaded_key = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, key: str):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed. Install a CUDA-enabled PyTorch build "
                "before using the local image generator."
            ) from exc
        from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
        from diffusers import EulerAncestralDiscreteScheduler

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch; CPU fallback is disabled.")
        with self.lock:
            if self.loaded_key == key and self.pipeline is not None:
                return self.pipeline
            self.unload()
            profile = LOCAL_MODELS[key]
            cls = StableDiffusionXLPipeline if profile.family == "sdxl" else StableDiffusionPipeline
            pipe = cls.from_pretrained(
                profile.model_id, revision=profile.revision,
                torch_dtype=torch.float16, use_safetensors=True,
            )
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            pipe.set_progress_bar_config(disable=True)
            self.pipeline = pipe
            self.loaded_key = key
            return pipe

    def generate(self, request: ImageRequest, *, strategy="batch",
                 progress_callback=None, image_callback=None):
        if strategy not in ("batch", "sequential"):
            raise ValueError("Generation strategy must be 'batch' or 'sequential'.")
        prompt, negative = compose(request)
        base_seed = request.seed or secrets.randbelow(2**31 - request.count)
        seeds = [base_seed + i for i in range(request.count)]
        with self.lock:
            pipe = self.load(request.model)
            import torch
            from diffusers import AutoPipelineForImage2Image

            if request.input_image:
                if self.image_pipeline is None:
                    self.image_pipeline = AutoPipelineForImage2Image.from_pipe(pipe)
                    self.image_pipeline.set_progress_bar_config(disable=True)
                pipe = self.image_pipeline
                with Image.open(request.input_image) as source:
                    source_image = source.convert("RGB").resize(
                        (request.width, request.height), Image.Resampling.LANCZOS
                    )
            images = []
            effective_batches = []
            offset, batch = 0, (1 if strategy == "sequential" else request.count)
            progress_fraction = 0.0
            torch.cuda.reset_peak_memory_stats()
            while offset < request.count:
                size = min(batch, request.count - offset)

                def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    nonlocal progress_fraction
                    step_total = (max(1, round(request.steps * request.strength))
                                  if request.input_image else request.steps)
                    image_fraction = min(1.0, (step_index + 1) / step_total)
                    fraction = (offset + size * image_fraction) / request.count
                    progress_fraction = max(progress_fraction, fraction)
                    if progress_callback:
                        progress_callback({
                            "fraction": min(0.99, progress_fraction),
                            "completed_images": offset,
                            "image_index": offset + 1,
                            "batch_start": offset,
                            "batch_size": size,
                            "image_fraction": image_fraction,
                            "count": request.count,
                            "step": step_index + 1,
                            "steps": step_total,
                            "strategy": strategy,
                        })
                    return callback_kwargs

                rng = [torch.Generator(device="cuda").manual_seed(s)
                       for s in seeds[offset:offset + size]]
                options = dict(prompt=prompt, negative_prompt=negative,
                               num_inference_steps=request.steps,
                               guidance_scale=request.guidance, generator=rng,
                               num_images_per_prompt=size,
                               callback_on_step_end=on_step_end)
                if request.input_image:
                    options.update(image=source_image, strength=request.strength)
                else:
                    options.update(width=request.width, height=request.height)
                try:
                    with torch.inference_mode():
                        result = pipe(**options)
                    images.extend(result.images)
                    if image_callback:
                        for index, image in enumerate(result.images, start=offset + 1):
                            image_callback(image, index, request.count)
                    effective_batches.append(size)
                    offset += size
                except torch.cuda.OutOfMemoryError as exc:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise RuntimeError("Out of GPU memory even with batch size 1. "
                                           "Reduce resolution or use a smaller model.") from exc
                    if progress_callback:
                        progress_callback({"retrying": True, "fraction": offset / request.count,
                                           "completed_images": offset, "batch_start": offset,
                                           "batch_size": size, "count": request.count,
                                           "strategy": strategy})
                    batch = max(1, size // 2)
            return images, {"prompt_sent": prompt, "negative_prompt_sent": negative,
                            "seed": base_seed, "per_image_seeds": seeds, "device": "cuda:0",
                            "effective_batches": effective_batches,
                            "generation_strategy": strategy,
                            "gpu": torch.cuda.get_device_name(0),
                            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
