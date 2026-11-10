"""Compare warmed-up serialized CUDA strategies without concurrent GPU threads."""
import argparse
import json
from statistics import median
from pathlib import Path

from .history import OUTPUT_ROOT
from .models import ImageRequest, LOCAL_MODELS
from .workflow import local_provider, run_generation


def main():
    parser = argparse.ArgumentParser(description="Benchmark local model CUDA throughput")
    parser.add_argument("--model", choices=LOCAL_MODELS, default="animagine")
    parser.add_argument("--prompt", default="1girl, green hair, shrine maiden, frog hat, lake")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--strategies", choices=("batch", "sequential"), nargs="+",
                        default=["batch", "sequential"],
                        help="Compare batched calls with serialized one-image calls.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Timed repetitions for each strategy and image count.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    profile = LOCAL_MODELS[args.model]
    provider = local_provider()
    print(json.dumps(provider.status(), indent=2))
    print("Loading model once (model download/load time excluded from measurements)...")
    provider.load(args.model)
    # Run a warm-up image so the first timed result excludes CUDA kernel startup.
    warmup = ImageRequest(args.prompt, model=args.model, width=profile.width,
                          height=profile.height, steps=profile.steps,
                          guidance=profile.guidance, seed=123456, count=1)
    print("Warming up CUDA with one unrecorded image...")
    provider.generate(warmup)
    for strategy in args.strategies:
        for count in args.counts:
            durations, rates = [], []
            for repeat in range(1, args.repeats + 1):
                request = ImageRequest(args.prompt, model=args.model, width=profile.width,
                                       height=profile.height, steps=profile.steps,
                                       guidance=profile.guidance, seed=123456 + repeat * count,
                                       count=count)
                directory = run_generation(request, args.output, strategy=strategy)
                data = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                duration = data["elapsed_seconds"]
                rate = count / duration if duration else 0
                durations.append(duration)
                rates.append(rate)
                print(json.dumps({
                    "type": "run",
                    "strategy": strategy,
                    "repeat": repeat,
                    "count": count,
                    "elapsed_seconds": duration,
                    "inference_seconds": data.get("inference_seconds"),
                    "save_seconds": data.get("save_seconds"),
                    "images_per_second": round(rate, 4),
                    "seconds_per_image": round(duration / count, 3),
                    "peak_vram_gb": data.get("peak_vram_gb"),
                    "effective_batches": data.get("effective_batches"),
                    "model": args.model,
                    "resolution": f"{request.width}x{request.height}",
                    "steps": request.steps,
                    "path": str(directory),
                }, ensure_ascii=False))
            print(json.dumps({
                "type": "summary", "strategy": strategy, "count": count,
                "repeats": args.repeats,
                "median_elapsed_seconds": round(median(durations), 3),
                "median_images_per_second": round(median(rates), 4),
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
