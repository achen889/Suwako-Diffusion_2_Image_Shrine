# Suwako Diffusion 2.0: Tink handoff specification

## Intent

Rebuild the user's September 2026 `suwako-art` CUDA image generator, not replace it
with a cloud-only app. The original `suwako-art.zip` profiles are the functional baseline.
User benchmark: CPU roughly **7 minutes per anime image**; prior CUDA setup **about 30
seconds for four images** described as four threads. Treat these as user-observed
benchmarks, not as proven results of this implementation.

## Implemented MVP

- **Default `local` backend:** PyTorch CUDA + Diffusers + Hugging Face.
- Profile `animagine`: `cagliostrolab/animagine-xl-4.0`, pinned SDXL revision.
- Profile `waifu`: `waifu-diffusion/wd-1-5-beta2`, pinned SD1.5 revision.
- Profile `anything_v4`: safetensors Diffusers community mirror, pinned revision.
- fp16 model weights; Euler Ancestral scheduler; local text-to-image and single-image img2img.
- One cached model, serial access via RLock; local four-image batch with OOM chunk retry.
- Prompt exact/enhanced modes, negative prompt, resolution, steps, CFG, seed, image count.
- Optional OpenAI `gpt-image-2.6-flare` and `gpt-image-2.6-sunburst` provider.
- CLI, Streamlit app, pywebview desktop wrapper, JSON metadata and persistent gallery.
- Physical image-library folders (Inbox and albums), SQLite catalog, tags, imports,
  paged search, batch organization, and verified legacy-output migration.
- Run manifests retain settings and link stable asset IDs; library images can be selected
  as img2img inputs and are recorded as source assets for the resulting run.
- Categorized JSON prompt presets, assembled/editable prompts, example recipes, and per-image progress.
- One-second elapsed timer while the serialized generation worker loads, generates, and saves.
- Timestamped console/file diagnostics and generation source selections in run metadata.
- Minimal tests for prompt handling, validation, saving, metadata (fake provider).

### User-observed CUDA baseline

On 2026-11-09, the user manually generated four 1024×1024 Animagine XL 4.0 images
at 28 steps and CFG 5.0 on an RTX 4080 Laptop GPU. The saved run reports 88.698
seconds, one batch of four, and 8.978 GB peak allocated VRAM. This is a baseline for
future comparisons, not a promise of the progress UI's performance.

## Important architectural decisions

1. **CUDA-first, no implicit CPU fallback.** Expose GPU model, VRAM, PyTorch CUDA build.
2. **Keep model providers independent.** The cloud backend is not a diffusion simulator:
   API has no sampler/CFG/seed even if the GUI has local controls.
3. **Benchmark actual throughput.** Four images in one CUDA batch is not automatically
   four parallel Python threads. Implement benchmarking for batch=1/2/4 and optionally
   four independent threads only as a controlled experiment, not MVP default.
4. **One GPU worker.** Streamlit reruns must never load two large CUDA pipelines or
   invoke the same pipeline simultaneously.
5. **Reproducibility.** Record exact HF model revision, package versions, GPU model,
   seeds per image, prompt composition and duration. Record actual batch/chunk sizes
   as part of the initial CUDA logging.
6. **Security.** `.env` is gitignored; no API key in metadata or logs. Bind Streamlit
   to loopback for desktop use. External booru fetching is out of MVP scope.

## Phase 1 acceptance tests (on user's NVIDIA PC)

- [ ] `python -m suwako.cli --status` displays CUDA available, model, VRAM.
- [ ] Cold launch downloads only the selected model, not both.
- [ ] Generate 4 images with Animagine and 4 with Waifu; no CPU inference.
- [ ] With explicit seed, output metadata records each per-image seed.
- [ ] GUI and CLI reuse the exact same generation path.
- [ ] Local img2img uses one user-provided image and adjustable strength.
- [ ] OOM fallback reduces batch size without discarding earlier generated images.
- [ ] OpenAI image generation and optional edit succeed only when a key is provided.
- [ ] Gallery displays images and settings after app restart.
- [ ] Desktop window exits cleanly and shuts down its Streamlit child process.
- [ ] Windows setup tested on clean venv. No `mew-ai`/`MewUtilities` required.

## Phase 2: performance and features

- Add cancellation and safe queue shutdown for an active generation worker.
- Benchmark four-image throughput (single batch vs repeated single inference),
  including SD1.5/SDXL, 512/768/1024 resolutions and peak VRAM.
- Add registered, validated Hugging Face model profiles and LoRA support.
- Add configurable prompt assistants (local LLM via Transformers/Ollama, optional OpenAI),
  borrowing provider-selection patterns from the old `labs/ai` language-model tests.
  Prompt enhancement must remain opt-in, reversible, and visible before generation.
- Add img2img/inpainting mask editor, ControlNet/IP-Adapter/reference image support.
- Restore the original side-by-side compare action (two local models, or optional paid API comparison).
- Regenerate from saved settings in the image library.
- Import/export reproducible prompt recipe JSON, image PNG metadata.
- Add a small local image-captioning/tagging provider based on `labs/ai/booru_test.py`,
  WITHOUT automatically crawling remote booru sites. Respect licensing/site terms.
- Add optional Gradio comparison demo but avoid parallel UI stacks in the main app.

## Legacy code reuse observations

The old CLI, Streamlit GUI, `history.py`, `presets.py` and `models.py` provide an
excellent baseline. Its actual CUDA work delegated to `mew.ai.DiffusionService`,
which is **not included** in the uploaded zip. The rebuilt MVP substitutes direct
Diffusers integration to avoid that missing, potentially unpublished dependency.

The user's `desktop.py` already implements robust localhost health-checking and
process shutdown; this was reused with a Suwako window title. Old image gallery
labs show the intended browsing workflow, but eager-loading every PIL file and
using a fixed three-column destructuring pattern should not be reproduced.

The booru lab is a research example, not production crawler: it contains
third-party utility imports, a global variable bug in captioning, `is` instead
of `==` for rating checks, and unbounded remote operations. Treat it as
inspiration for an **opt-in, rate-limited** image-reference importer later.

## Known limitations / risks

- CUDA and API generation have not been run in this container: no NVIDIA GPU,
  downloaded models, or API key is available here. The code is a testable scaffold.
- Confirm HF revision hashes, licenses and any gated model permissions on first run.
- Changing Diffusers/Transformers/Torch versions can require compatibility tweaks.
- Big SDXL 4-image batches may be VRAM constrained; fallback is important.
- Multi-reference editing is only present on OpenAI provider in MVP.
