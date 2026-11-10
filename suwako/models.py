"""Provider-neutral input settings, with pinned local Hugging Face profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    key: str
    name: str
    model_id: str
    revision: str
    family: str
    width: int
    height: int
    steps: int
    guidance: float
    suffix: str
    negative: str
    prefix: str = ""


LOCAL_MODELS = {
    "animagine": ModelProfile(
        "animagine", "Animagine XL 4.0", "cagliostrolab/animagine-xl-4.0",
        "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96", "sdxl", 1024, 1024, 28, 5.5,
        "masterpiece, high score, great score, absurdres",
        "lowres, bad anatomy, bad hands, text, error, cropped, low quality, "
        "low score, bad score, signature, watermark, blurry",
    ),
    "waifu": ModelProfile(
        "waifu", "Waifu Diffusion 1.5 Beta 2", "waifu-diffusion/wd-1-5-beta2",
        "552da11", "sd15", 512, 512, 25, 7.0,
        "", "worst quality, low quality, lowres, bad anatomy, bad hands, text, "
        "error, cropped, jpeg artifacts, signature, watermark, blurry",
        "masterpiece, best quality",
    ),
    "anything_v4": ModelProfile(
        "anything_v4", "Anything V4 (community Diffusers mirror)",
        "Yntec/AnythingV4", "c5ee980", "sd15", 768, 768, 28, 7.0,
        "masterpiece, best quality, detailed anime illustration",
        "worst quality, low quality, lowres, bad anatomy, bad hands, text, "
        "error, cropped, jpeg artifacts, signature, watermark, blurry",
    ),
}
OPENAI_MODELS = ("gpt-image-2.6-flare", "gpt-image-2.6-sunburst")


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    provider: str = "local"
    model: str = "animagine"
    negative_prompt: str = ""
    mode: str = "enhanced"
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 5.5
    seed: int = 0
    count: int = 4
    input_image: Path | None = None
    strength: float = 0.65
    reference_images: tuple[Path, ...] = ()
    quality: str = "medium"

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("A prompt is required.")
        if self.provider not in ("local", "openai"):
            raise ValueError("Provider must be 'local' or 'openai'.")
        choices = LOCAL_MODELS if self.provider == "local" else OPENAI_MODELS
        if self.model not in choices:
            raise ValueError(f"Unknown {self.provider} model: {self.model}")
        if self.mode not in ("exact", "enhanced"):
            raise ValueError("Prompt mode must be exact or enhanced.")
        if not 1 <= self.count <= 4:
            raise ValueError("MVP batch size must be 1–4.")
        if self.provider == "local":
            if (self.width % 64 or self.height % 64 or
                    not (256 <= self.width <= 2048 and 256 <= self.height <= 2048)):
                raise ValueError("Local image sizes must be 256–2048 and divisible by 64.")
            if not 1 <= self.steps <= 100 or not 0 <= self.guidance <= 30:
                raise ValueError("Steps or guidance out of range.")
            if self.reference_images:
                raise ValueError("Local backend supports one input image (img2img), not multiple references yet.")
        if self.provider == "openai" and self.quality not in ("low", "medium", "high"):
            raise ValueError("Unsupported OpenAI quality setting.")
        if not 0 <= self.seed < 2**31:
            raise ValueError("Seed must be 0 (random) or a 31-bit positive integer.")
        if not 0 < self.strength <= 1:
            raise ValueError("Img2img strength must be in (0, 1].")
        for path in ([self.input_image] if self.input_image else []) + list(self.reference_images):
            if not path.is_file():
                raise FileNotFoundError(f"Missing reference image: {path}")
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                raise ValueError(f"Unsupported reference image type: {path}")
