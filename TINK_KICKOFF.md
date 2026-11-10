# Tink's first mission | Suwako Diffusion 2.0 🐸🎨

Hello, Tink! Welcome back to the little frog shrine of GPU-powered art.

We're reviving **Suwako Diffusion**, an old Python anime-image generator, with better architecture and better agent tooling. The original project worked with Hugging Face models accelerated by NVIDIA CUDA. The user remembers about **four CUDA-generated images in 30 seconds** on their machine, versus roughly **seven minutes for one CPU image**. We want to find out what today's code can actually achieve, not assume those numbers are still reproducible.

**This is a CUDA-first local application.** Don't turn it into an OpenAI-only frontend! OpenAI image generation/editing is a handy optional second provider, never the default or an automatic fallback. Avoid paid API calls unless explicitly authorized.

## Where to start

Read `AGENTS.md`, `README.md`, `TINK_SPEC.md`, and scan the supplied `references/` files. You have a working scaffold: Streamlit GUI, pywebview Windows launcher, CLI, separate CUDA/OpenAI providers, image-saving/history, and mock-backed unit tests. The original `suwako-art.zip` is present as a reference; its `mew.ai.DiffusionService` isn't bundled, so use the independent Diffusers implementation rather than depending on that package.

## Your first assignment: restore the GPU pipeline

1. Inspect the repository and run the existing tests. Report the baseline before changing code.
2. Diagnose the actual local machine: Python, NVIDIA GPU, VRAM, CUDA-enabled PyTorch, Diffusers, Transformers, compatible packages, and Hugging Face access/model revisions.
3. Make **one-image** local text-to-image generation work with Animagine XL 4.0; then do a **four-image** run. Repeat with Waifu Diffusion 1.5 if practical.
4. Benchmark batches of 1, 2, and 4 and record elapsed time, peak VRAM, effective batch sizes, resolution, steps, model, and software environment. The user's old 'four threads' recollection is a clue to investigate, not a license to fire four concurrent pipeline calls at one GPU. Batch first, benchmark concurrency later if justified.
5. Confirm saved PNGs plus metadata, gallery reopening, and the Streamlit/pywebview launch path. Fix the smallest issues that prevent a comfortable end-to-end session.
6. Give me a brief checkpoint report: what runs, what doesn't, benchmark numbers, files changed, tests run, and your recommended next task.

## Working style

Be cheerful, direct, and technically skeptical. Solve tangible problems, not speculative rewrites. Keep changes small enough to review, preserve the existing source, and ask before incurring API costs or deleting outputs. Treat lab scripts as historical inspiration; never mass-import them. Keep the desktop UI simple until the GPU backend earns its frog badge.

**Victory condition:** I can open Suwako, enter a prompt, select an anime model, click Generate, get up to four CUDA images, and find them again in the gallery. Once that works, we can have fun with prompt presets, reference images, LoRAs, gallery search, and optional OpenAI comparisons.

Let's get the frog shrine rendering again. 🐸
