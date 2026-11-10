"""python -m suwako.cli 'prompt' --count 4 --model animagine"""
import argparse
from pathlib import Path
import sys

from .history import OUTPUT_ROOT
from .models import ImageRequest, LOCAL_MODELS, OPENAI_MODELS
from .workflow import run_generation, local_provider


def main() -> int:
    parser = argparse.ArgumentParser(description="Suwako Studio: CUDA-first image generation")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--negative", default="")
    parser.add_argument("--mode", choices=("exact", "enhanced"), default="enhanced")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--quality", choices=("low", "medium", "high"),
                        default="medium")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--check-cuda", action="store_true",
                        help="Report the active Python and exit non-zero unless CUDA is available.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.status or args.check_cuda:
        import json
        try:
            status = local_provider().status()
        except Exception as error:
            status = {
                "cuda_available": False,
                "diagnostic": f"CUDA status check failed: {type(error).__name__}: {error}",
            }
        status["python"] = sys.executable
        print(json.dumps(status, indent=2))
        return 1 if args.check_cuda and not status.get("cuda_available") else 0
    if not args.prompt:
        parser.error("A prompt is required unless --status is used.")
    model = args.model or ("animagine" if args.provider == "local" else OPENAI_MODELS[0])
    if args.provider == "local":
        if model not in LOCAL_MODELS:
            parser.error(f"Unknown local model. Select from: {', '.join(LOCAL_MODELS)}")
        profile = LOCAL_MODELS[model]
    else:
        profile = None
    request = ImageRequest(
        prompt=args.prompt, provider=args.provider, model=model,
        negative_prompt=args.negative, mode=args.mode,
        width=args.width or (profile.width if profile else 1024),
        height=args.height or (profile.height if profile else 1024),
        steps=args.steps or (profile.steps if profile else 28),
        guidance=args.guidance if args.guidance is not None else
                 (profile.guidance if profile else 5.0),
        seed=args.seed, count=args.count if args.count is not None else (4 if args.provider == "local" else 1), input_image=args.input_image,
        reference_images=tuple(args.reference), strength=args.strength,
        quality=args.quality,
    )
    from dotenv import load_dotenv
    load_dotenv()
    directory = run_generation(request, args.output)
    print(f"Saved to {directory.resolve()}")
    for image in sorted(directory.glob("image_*.png")):
        print(image.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
