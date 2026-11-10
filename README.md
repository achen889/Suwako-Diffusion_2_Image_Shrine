# Suwako Diffusion 2.0 MVP

**Tink handoff:** start with [AGENTS.md](AGENTS.md) (standing engineering instructions) and [TINK_KICKOFF.md](TINK_KICKOFF.md) (the first-session mission). `references/` contains a safe, curated subset of original examples and the legacy Suwako project. The full `labs.zip` is not bundled because it includes a potential hard-coded credential.

Local CUDA / Hugging Face generation **by default**, with optional GPT Image API.
The service is shared by a Streamlit web UI, a pywebview desktop wrapper, and a CLI.

## Your image shrine

Suwako Diffusion is a local-first studio for creating and curating anime images. Its
Review Gallery is a personal image shrine: browse generations by character, inspect
their prompts and settings, save favorites to a `best` collection, download images,
and remove individual images or selected batches. Generated images and uploads stay
in local folders on your computer.

## Windows setup (Python 3.11 or 3.12 recommended)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a **CUDA-capable PyTorch** version matched to your NVIDIA GPU and drivers using
https://pytorch.org/get-started/locally/ . Verify with
`python -c "import torch; print(torch.cuda.is_available())"` (must print `True`).
Then install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe -m suwako.cli --check-cuda
run_studio.bat
```

The launchers always use `.venv\Scripts\python.exe`; they stop with a diagnostic if
that environment is missing or cannot access CUDA. You can launch Streamlit directly
with `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1`.
For a desktop window instead, run `run_desktop.bat` (requires WebView2 on Windows).
Running `python desktop.py` also re-launches through the project venv.
For the command line:

```powershell
.\.venv\Scripts\python.exe -m suwako.cli "1girl, green hair, shrine maiden, frog hat" --model animagine --count 4
.\.venv\Scripts\python.exe -m suwako.cli "forest witch" --model waifu --count 4 --width 512 --height 512
.\.venv\Scripts\python.exe -m suwako.cli "frog shrine by a pond" --model animagine --input-image source.png --strength 0.6
```

## Optional OpenAI backend

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (do not commit the key).

```powershell
.\.venv\Scripts\python.exe -m suwako.cli "Illustrate a small shrine on a misty pond" --provider openai --model gpt-image-2.6-flare --count 1
.\.venv\Scripts\python.exe -m suwako.cli "Keep the original subject; add evening lanterns" --provider openai --input-image source.png
```

API billing is separate from ChatGPT subscriptions. No OpenAI key is needed locally. The OpenAI provider defaults to **one** image to avoid accidental multi-image API spending. Using it transmits the prompt and reference images to OpenAI.

## GPU performance and safety

- Default: a **single four-image Diffusers batch**, one pipeline loaded at a time. Cloud defaults to one image to avoid unintended API spend.
- The UI offers serialized one-at-a-time generation with a live preview after each completed image.
- If CUDA runs out of memory, the batch automatically retries in chunks of 2 or 1.
- float16 weights, modern PyTorch attention, cached Hugging Face downloads.
- CUDA kernel warm-up is performed before benchmark timings (not before everyday generation).
- CUDA is **required** for local generation; there is no silent seven-minute CPU fallback.
- Seeds are deterministic inputs but identical pixels across package/hardware changes are not guaranteed.
- The very first run downloads model weights. Animagine is a large SDXL model.
- Only one generating worker per CUDA device. Multiple CPU threads do not guarantee faster inference.
- Model licenses and any local source/reference image rights remain the user's responsibility.

Images and `metadata.json` live under `outputs/<character>/<timestamp>_<prompt>_<id>/`.
The Streamlit **Review gallery** tab shows saved images, supports downloading and deleting
individual or checked batches of images, and can open a run's folder in Explorer. Thumbs-up
images go into a `best` folder by default; the gallery can rename that collection folder
across characters. Character folders come from
the selected character preset; runs without one (including CLI runs) are stored under
`outputs/uncategorized/`. Older runs directly under `outputs/` remain visible. Uploaded
input files are cached in `uploads/`.
No image files or Hugging Face model weights are bundled in this project.

Prompt components and recipes are editable JSON files under `data/presets/`.
They are organized into `characters/`, `clothing/`, `poses/`, `backgrounds/`,
`other/`, and `recipes/`; `defaults.json` contains the standard negative prompt.
The Generate tab shows a saved-preset text editor directly below the selected
character, clothing, pose, or background preset. Edit its comma-separated tags and
save to write the changes back to that preset's JSON file. Each generated run records the selected
component groups and tags in its metadata. Reference-image inputs and img2img strength
are grouped in the collapsed **Image-to-image options** section; uploaded references
are previewed there before generation.
Application diagnostics are written continuously to per-process timestamped files in `logs/`
and also sent to the launcher console when one is available. Logs, uploads, and
generated output are excluded from Git.

<details>
<summary>Changelog</summary>

### Unreleased

- Added a collapsible image gallery with per-image review actions and batch deletion.
- Added a renameable favorites folder, named `best` by default.
- Added editable character, clothing, pose, background, and recipe presets.
- Recorded model-specific prompt limits and generation observations in `docs/model-notes/`.

See [CHANGELOG.md](CHANGELOG.md) for the full project history.
</details>

## Upload with Git

This workspace is initialized as a local Git repository on `main`. Before the first
push, create an empty GitHub repository and configure its URL as `origin`:

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git add -A
git status --short
git diff --cached --stat
git commit -m "Prepare Suwako Diffusion MVP"
git push -u origin main
```

Review the staged files before committing. `.gitignore` excludes environment files,
image/video files, generated outputs, uploads, logs, model weights, caches, and temporary
files. Keep `.env.example`; it contains empty placeholders, not credentials.

For later uploads, run `upload_to_github.bat` from Explorer or PowerShell. It stages
non-ignored changes, shows the staged-file list, asks before committing, then pushes the
current branch to `origin`. Pass a commit message as the first argument to skip its
message prompt, for example `upload_to_github.bat "Update prompt presets"`.

## Run a GPU throughput comparison

After your local model is cached and confirmed working:

```powershell
.\.venv\Scripts\python.exe -m suwako.benchmark --model animagine --counts 1 2 4
```

By default this warms the selected model and records 1-, 2- and 4-image runs for both
batched and serialized one-at-a-time generation. Each JSON result reports inference
time, image-save time, total time, images per second, peak VRAM and any out-of-memory
batch splitting. It prints per-run results and median elapsed time / throughput.
Default settings measure 15 images across 11 pipeline calls, including the one-image
warm-up. Add `--repeats 3` for three timed repetitions per case. Use
`--strategies batch` to run only the three batch measurements. It deliberately does
**not** run independent threads against the same pipeline, since that is not a safe
default for a single GPU.

Available local profiles include Animagine XL 4.0, Waifu Diffusion 1.5 Beta 2, and
Anything V4 from the [Yntec Diffusers mirror](https://huggingface.co/Yntec/AnythingV4),
listed under its CreativeML OpenRAIL-M license. The loader accepts safetensors only.
The [legacy Pastel Mix mirror](https://huggingface.co/LibreSD/Andite/blob/main/pastelmix.safetensors)
currently provides a single checkpoint rather than the Diffusers directory layout this
app loads, so it is not listed as a ready-to-use model.

## Tests and limits

```powershell
python -m unittest discover -s tests -v
```

Tests use a fake provider, so they can run without GPU or an API key. **Live CUDA and OpenAI
inference must still be smoke-tested on your machine.** The MVP does not yet provide
generation cancellation, LoRA management, inpainting masks, arbitrary checkpoints,
or multiple local image references.

See [TINK_SPEC.md](TINK_SPEC.md) for implementation priorities, acceptance criteria,
and notes from the original September 2026 `suwako-art` project.
