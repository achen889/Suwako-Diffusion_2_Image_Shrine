"""Run using: streamlit run app.py"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import streamlit as st

from suwako.history import (
    OUTPUT_ROOT,
    best_folder_name,
    discard_image,
    discard_images,
    keep_best,
    recent_runs,
    rename_best_folder,
)
from suwako.models import ImageRequest, LOCAL_MODELS, OPENAI_MODELS
from suwako.prompts import (
    PROMPT_CONFIG_PATH,
    compose,
    assemble_selected_tags,
    join_prompt_tags,
    load_prompt_config,
    split_prompt_tags,
    tags_for_components,
)
from suwako.formatting import format_duration
from suwako.logging_utils import configure_logging
from suwako.workflow import run_generation, local_provider

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
LOGGER, SESSION_LOG = configure_logging()


def selected_components(category: str):
    selected = st.session_state.get(f"{category}_components", [])
    return [selected] if isinstance(selected, str) else list(selected)


def rebuild_prompt_from_categories(reset_recipe=False):
    selections = {
        category: st.session_state.get(f"{category}_tags", [])
        for category in ("characters", "clothing", "poses", "backgrounds", "other")
    }
    st.session_state["prompt_input"] = assemble_selected_tags(selections)
    st.session_state["_manual_prompt_override"] = False
    if reset_recipe:
        st.session_state["recipe_selection"] = "custom"


def sync_prompt_from_category_tags():
    rebuild_prompt_from_categories(reset_recipe=True)


def update_category_tags(config: dict, category: str):
    st.session_state[f"{category}_tags"] = tags_for_components(
        config["categories"][category], selected_components(category)
    )
    rebuild_prompt_from_categories(reset_recipe=True)


def apply_recipe(config: dict):
    recipe_key = st.session_state.get("recipe_selection", "custom")
    recipe = next((item for item in config["recipes"] if item["key"] == recipe_key), None)
    if recipe is None:
        return
    for category in ("characters", "clothing", "poses", "backgrounds", "other"):
        selected = recipe.get("components", {}).get(category, [])
        st.session_state[f"{category}_components"] = selected
        st.session_state[f"{category}_tags"] = tags_for_components(
            config["categories"][category], selected
        )
    rebuild_prompt_from_categories()


def mark_prompt_as_manual():
    st.session_state["_manual_prompt_override"] = True
    st.session_state["recipe_selection"] = "custom"


def sync_negative_tags_from_text():
    st.session_state["negative_prompt_tags"] = split_prompt_tags(
        st.session_state.get("negative_prompt_input", "")
    )


def sync_negative_text_from_tags():
    st.session_state["negative_prompt_input"] = join_prompt_tags(
        st.session_state.get("negative_prompt_tags", [])
    )


def initialize_prompt_state(config: dict):
    if st.session_state.get("_categorized_presets_initialized"):
        return
    initial = {
        "characters": ["suwako_moriya"],
        "clothing": ["suwako_clothing"],
        "poses": [],
        "backgrounds": ["suwako_shrine_pond"],
        "other": [],
    }
    for category, selected in initial.items():
        existing = {item["key"] for item in config["categories"][category]}
        selected = [key for key in selected if key in existing]
        st.session_state[f"{category}_components"] = selected
        st.session_state[f"{category}_tags"] = tags_for_components(
            config["categories"][category], selected
        )
    st.session_state["prompt_input"] = assemble_selected_tags({
        category: st.session_state[f"{category}_tags"]
        for category in config["categories"]
    })
    st.session_state["_manual_prompt_override"] = False
    st.session_state["negative_prompt_input"] = config["default_negative_prompt"]
    st.session_state["negative_prompt_tags"] = split_prompt_tags(
        config["default_negative_prompt"]
    )
    st.session_state["recipe_selection"] = "custom"
    st.session_state["_categorized_presets_initialized"] = True


def store_upload(upload) -> Path:
    payload = upload.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    suffix = Path(upload.name).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("Only PNG, JPG, JPEG and WebP image references are supported.")
    UPLOADS.mkdir(exist_ok=True)
    path = UPLOADS / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(payload)
    return path


def save_preset_tags(category: str, preset_key: str, tags: list[str]) -> Path:
    """Replace one preset's tags while preserving its other JSON fields."""
    category_dir = (PROMPT_CONFIG_PATH / category).resolve()
    matches = []
    for path in category_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("key") == preset_key:
            matches.append((path, data))
    if len(matches) != 1:
        raise ValueError(f"Expected one saved preset for {preset_key!r} in {category}.")
    path, data = matches[0]
    data["tags"] = tags
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    temp_path.replace(path)
    return path


def st_gallery_fancy():
    """Render saved images as review cards with explicit local file actions."""
    output_root = OUTPUT_ROOT.resolve()

    current_best_folder = best_folder_name(OUTPUT_ROOT)
    with st.expander("Best image folder", expanded=False):
        st.caption(f"👍 Best images are stored in a folder named “{current_best_folder}”.")
        requested_best_folder = st.text_input(
            "Folder name for 👍 images",
            value=current_best_folder,
            key=f"gallery-best-folder-name-{current_best_folder}",
        )
        if st.button("Rename best folder", key="gallery-rename-best-folder"):
            try:
                rename_best_folder(requested_best_folder, OUTPUT_ROOT)
            except (OSError, ValueError) as exc:
                st.error(f"Could not rename best folder: {exc}")
            else:
                st.rerun()

    runs = recent_runs(OUTPUT_ROOT, limit=200)

    def is_best_directory(directory, data):
        return bool(data.get("best_of")) or directory.parent.name.casefold() in {
            "best", current_best_folder.casefold(),
        }

    def folder_name(directory, data=None):
        data = data or {}
        if is_best_directory(directory, data) and directory.parent.parent.resolve() != output_root:
            return directory.parent.parent.name + " · " + directory.parent.name
        return directory.parent.name if directory.parent.resolve() != output_root else "legacy runs"

    images = []
    for directory, data in runs:
        for name in data.get("images", []):
            if not isinstance(name, str) or Path(name).name != name:
                continue
            file = directory / name
            if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if file.is_file() and file.resolve().is_relative_to(output_root):
                images.append((directory, data, name))
    if not images:
        st.info("No saved images yet. Generate images to see them here.")
        return

    character_folders = sorted({
        folder_name(directory, data) for directory, data, _ in images
    })
    selected_character = st.selectbox(
        "Character folder", ["All characters", *character_folders],
        key="gallery_character_filter",
    )
    if selected_character != "All characters":
        images = [item for item in images
                  if folder_name(item[0], item[1]) == selected_character]
    st.caption(f"{len(images)} saved image(s) · stored under {OUTPUT_ROOT}")

    def selection_key(directory, name):
        digest = hashlib.sha1(str(directory.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"gallery-select-{digest}-{name}"

    selected_images = [
        item for item in images
        if st.session_state.get(selection_key(item[0], item[2]), False)
    ]
    if st.button(
        f"Delete selected images ({len(selected_images)})",
        key="gallery-delete-selected",
        disabled=not selected_images,
        help="Delete only the checked images. Other images and run records are kept.",
    ):
        try:
            deleted = discard_images(
                [(directory, name) for directory, _, name in selected_images], OUTPUT_ROOT
            )
        except (OSError, ValueError) as exc:
            st.error(f"Could not delete selected images: {exc}")
        else:
            for directory, _, name in selected_images:
                st.session_state.pop(selection_key(directory, name), None)
            st.success(f"Deleted {deleted} selected image(s).")
            st.rerun()

    columns_count = 4
    for offset in range(0, len(images), columns_count):
        columns = st.columns(columns_count)
        for column, (directory, data, name) in zip(columns, images[offset:offset + columns_count]):
            file = directory / name
            with column:
                with st.container(border=True):
                    st.image(str(file), use_container_width=True)
                    st.checkbox("Select for deletion", key=selection_key(directory, name))
                    st.caption(f"{folder_name(directory, data)} · {directory.name}")
                    st.caption(f"{data.get('provider', '?')} / {data.get('model', '?')}")
                    best_dir = is_best_directory(directory, data)
                    rating_columns = st.columns(2)
                    with rating_columns[0]:
                        if st.button(
                            "👍 Best" if best_dir else f"👍 Keep {current_best_folder}",
                            key=f"gallery-best-{directory.name}-{name}",
                            disabled=best_dir, use_container_width=True,
                            help="Move this image into the character's best folder.",
                        ):
                            try:
                                keep_best(directory, name, OUTPUT_ROOT)
                            except (OSError, ValueError) as exc:
                                st.error(f"Could not move image to best: {exc}")
                            else:
                                st.rerun()
                    with rating_columns[1]:
                        if st.button(
                            "👎 Delete", key=f"gallery-discard-{directory.name}-{name}",
                            use_container_width=True,
                            help="Delete this image from disk immediately.",
                        ):
                            try:
                                discard_image(directory, name, OUTPUT_ROOT)
                            except (OSError, ValueError) as exc:
                                st.error(f"Could not delete image: {exc}")
                            else:
                                st.rerun()
                    st.download_button(
                        "Save image", file.read_bytes(), file_name=name,
                        key=f"gallery-save-{directory.name}-{name}",
                        use_container_width=True,
                    )
                    if st.button(
                        "Open folder", key=f"gallery-open-{directory.name}-{name}",
                        use_container_width=True,
                    ):
                        if hasattr(os, "startfile"):
                            os.startfile(str(directory))
                        else:
                            st.info(f"Open this folder on your computer: {directory}")

    st.subheader("Generation records")
    for directory, data in runs:
        with st.expander(
            f"{folder_name(directory, data)} / {directory.name} · {data.get('model', '?')}"
        ):
            st.caption(str(directory))
            if st.button("Open run folder", key=f"gallery-run-open-{directory.name}"):
                if hasattr(os, "startfile"):
                    os.startfile(str(directory))
            st.json(data, expanded=False)


def st_latest_run_curation():
    """Keep the just-generated batch reviewable on the Generate tab."""
    last_run = st.session_state.get("last_run")
    if not last_run:
        return
    directory = Path(last_run)
    output_root = OUTPUT_ROOT.resolve()
    if not directory.is_dir() or not directory.resolve().is_relative_to(output_root):
        return
    try:
        data = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    images = [
        name for name in data.get("images", [])
        if isinstance(name, str) and Path(name).name == name
        and (directory / name).is_file()
        and (directory / name).resolve().is_relative_to(output_root)
    ]
    if not images:
        return
    with st.expander("Curate latest batch", expanded=False):
        columns = st.columns(min(4, len(images)))
        for column, name in zip(columns, images):
            file = directory / name
            with column:
                st.image(str(file), use_container_width=True)
                actions = st.columns(2)
                with actions[0]:
                    if st.button(
                                 f"👍 Keep {best_folder_name(OUTPUT_ROOT)}",
                                 key=f"latest-best-{directory.name}-{name}",
                                 use_container_width=True):
                        try:
                            keep_best(directory, name, OUTPUT_ROOT)
                        except (OSError, ValueError) as exc:
                            st.error(f"Could not move image to best: {exc}")
                        else:
                            st.rerun()
                with actions[1]:
                    if st.button("👎 Delete", key=f"latest-delete-{directory.name}-{name}",
                                 help="Delete this image from disk immediately.",
                                 use_container_width=True):
                        try:
                            discard_image(directory, name, OUTPUT_ROOT)
                        except (OSError, ValueError) as exc:
                            st.error(f"Could not delete image: {exc}")
                        else:
                            st.rerun()


def main():
    st.set_page_config(page_title="Suwako Diffusion", page_icon="🎨", layout="wide")
    st.title("Suwako Diffusion 🎨")
    st.caption("CUDA-first anime image studio | optional OpenAI image provider")
    prompt_config = load_prompt_config()
    initialize_prompt_state(prompt_config)
    refresh_category = st.session_state.pop("_refresh_saved_preset_category", None)
    if refresh_category:
        st.session_state[f"{refresh_category}_tags"] = tags_for_components(
            prompt_config["categories"][refresh_category],
            selected_components(refresh_category),
        )
        rebuild_prompt_from_categories()
    saved_preset_notice = st.session_state.pop("_saved_preset_notice", None)
    if not st.session_state.get("_logging_announced"):
        LOGGER.info("Studio opened; preset directory: %s; log file: %s",
                    PROMPT_CONFIG_PATH, SESSION_LOG)
        st.session_state["_logging_announced"] = True

    with st.sidebar:
        provider = st.radio("Engine", ("local", "openai"), format_func=lambda x:
                            "Local CUDA + Hugging Face" if x == "local" else "OpenAI Images API")
        options = LOCAL_MODELS if provider == "local" else OPENAI_MODELS
        model = st.selectbox("Model", list(options), format_func=lambda key:
                             LOCAL_MODELS[key].name if provider == "local" else key)
        if provider == "local":
            try:
                info = local_provider().status()
                if info["cuda_available"]:
                    st.success(f"{info['gpu']} | {info['vram_gb']} GB VRAM")
                else:
                    diagnostic = info.get("diagnostic") or "CUDA is unavailable; CPU generation is disabled."
                    st.error(f"{diagnostic} Active Python: {sys.executable}")
            except ImportError:
                st.error(f"Install CUDA PyTorch in this interpreter before local generation: {sys.executable}")
        else:
            st.info("Cloud generation sends prompts and uploaded reference images to OpenAI. Set OPENAI_API_KEY in your environment or .env file.")

    create_tab, review_tab = st.tabs(("Generate", "Review gallery"))
    with create_tab:
        if saved_preset_notice:
            st.success(saved_preset_notice)
        st.caption(f"Composable prompt files: {PROMPT_CONFIG_PATH}")
        tab_labels = ("Character", "Clothing", "Pose", "Background", "Other")
        categories = ("characters", "clothing", "poses", "backgrounds", "other")
        tabs = st.tabs(tab_labels)
        for tab, category, label in zip(tabs, categories, tab_labels):
            with tab:
                entries = prompt_config["categories"][category]
                names = {item["key"]: item["name"] for item in entries}
                all_tags = list(dict.fromkeys(
                    tag for item in entries for tag in item["tags"]
                ))
                if category == "other" and prompt_config["recipes"]:
                    recipes = {item["key"]: item["name"] for item in prompt_config["recipes"]}
                    st.selectbox(
                        "Load a legacy example combination",
                        ["custom", *recipes],
                        format_func=lambda key: "Custom composition" if key == "custom" else recipes[key],
                        key="recipe_selection",
                        on_change=apply_recipe,
                        args=(prompt_config,),
                    )
                st.multiselect(
                    f"{label} preset groups",
                    list(names),
                    format_func=lambda key, lookup=names: lookup.get(key, key),
                    key=f"{category}_components",
                    on_change=update_category_tags,
                    args=(prompt_config, category),
                    help="Select a saved group, then add or remove individual tags below.",
                )
                if category in ("characters", "clothing", "poses", "backgrounds") and entries:
                    selected_keys = [
                        key for key in selected_components(category) if key in names
                    ]
                    if selected_keys:
                        preset_key = selected_keys[0]
                        if len(selected_keys) > 1:
                            preset_key = st.selectbox(
                                f"{label} preset to edit",
                                selected_keys,
                                format_func=lambda key, lookup=names: lookup.get(key, key),
                                key=f"preset-file-select-{category}-{'-'.join(selected_keys)}",
                            )
                        preset = next(item for item in entries if item["key"] == preset_key)
                        editor_revision = hashlib.sha1(
                            join_prompt_tags(preset["tags"]).encode("utf-8")
                        ).hexdigest()[:10]
                        edited_tags = st.text_area(
                            f"Edit {names[preset_key]} prompt (comma-separated tags)",
                            value=join_prompt_tags(preset["tags"]),
                            key=f"preset-file-editor-{category}-{preset_key}-{editor_revision}",
                            height=100,
                        )
                        if st.button(
                            f"Save {names[preset_key]} prompt to preset file",
                            key=f"preset-file-save-{category}-{preset_key}",
                        ):
                            try:
                                saved_path = save_preset_tags(
                                    category, preset_key, split_prompt_tags(edited_tags)
                                )
                                st.session_state["_refresh_saved_preset_category"] = category
                                st.session_state["_saved_preset_notice"] = (
                                    f"Saved {names[preset_key]} to {saved_path}."
                                )
                                st.rerun()
                            except (OSError, ValueError, json.JSONDecodeError) as exc:
                                st.error(f"Could not save this preset: {exc}")
                    else:
                        st.caption("Select a preset group above to edit its saved prompt.")
                st.multiselect(
                    f"{label} tags — deselect to remove, type to add",
                    all_tags,
                    key=f"{category}_tags",
                    accept_new_options=True,
                    on_change=sync_prompt_from_category_tags,
                )
    
        prompt = st.text_area(
            "Final assembled prompt (editable)", height=130,
            placeholder="Select prompt groups above or enter your prompt...",
            key="prompt_input",
            on_change=mark_prompt_as_manual,
        )
        if st.session_state.get("_manual_prompt_override"):
            st.caption("Manual prompt edits are active. Changing a tab selection rebuilds this box from those categories.")
        with st.expander("Negative prompt", expanded=False):
            negative = st.text_area(
                "Things to avoid", height=100,
                key="negative_prompt_input",
                on_change=sync_negative_tags_from_text,
            )
            st.multiselect(
                "Negative tags — deselect to remove, type to add",
                split_prompt_tags(st.session_state["negative_prompt_input"]),
                key="negative_prompt_tags",
                accept_new_options=True,
                on_change=sync_negative_text_from_tags,
            )
        mode = st.radio("Prompt processing", ("enhanced", "exact"), horizontal=True)
    
        with st.expander("Image-to-image options", expanded=False):
            reference = st.file_uploader(
                "Input image (img2img for local / image edit for OpenAI)",
                type=("png", "jpg", "jpeg", "webp"),
                key="input_reference_image",
            )
            if reference is not None:
                st.image(
                    reference.getvalue(), caption=f"Reference image: {reference.name}",
                    use_container_width=True,
                )
            if provider == "local":
                strength = st.slider(
                    "Img2img strength", 0.05, 1.0, 0.65, 0.05,
                    disabled=reference is None,
                    help="Used only when an input image is uploaded.",
                )
                extras = []
            else:
                strength = 0.65
                extras = st.file_uploader(
                    "Additional OpenAI reference images", accept_multiple_files=True,
                    type=("png", "jpg", "jpeg", "webp"),
                    key="additional_reference_images",
                ) or []
                for extra in extras:
                    st.image(extra.getvalue(), caption=extra.name, use_container_width=True)
    
        count = st.slider("Images per request", 1, 4, 4 if provider == "local" else 1)
        with st.expander("Advanced generation settings", expanded=False):
            if provider == "local":
                size_opts = ("512x512", "768x768", "1024x1024", "1024x1536", "1536x1024")
                size = st.selectbox("Resolution", size_opts, index=2, key="resolution-local")
                steps = st.slider("Sampling steps", 5, 60, 28)
                guidance = st.slider("CFG / Guidance", 0.0, 15.0, 5.5, 0.5)
                seed = st.number_input("Seed (0 = random)", 0, 2**31 - 1, 0)
                strategy_label = st.radio(
                    "GPU execution", ("Batched", "One at a time (live previews)"),
                    help="Both modes use one GPU pipeline at a time. Batched mode is usually faster; one-at-a-time mode displays each image as soon as it finishes.",
                    horizontal=True,
                )
                strategy = "sequential" if strategy_label.startswith("One at a time") else "batch"
                st.caption("CUDA work stays serialized. Batched mode halves the batch automatically after an out-of-memory error.")
                quality = "medium"
            else:
                size = st.selectbox("Resolution", ("1024x1024", "1536x1024", "1024x1536"))
                quality = st.selectbox("API quality", ("low", "medium", "high"),
                                       index=1)
                steps, guidance, seed = 28, 5.0, 0
                strategy = "batch"
                st.caption("API has no diffusion sampler, CFG or reproducible seed controls.")
        width, height = map(int, size.split("x"))
    
        preview_prompt, preview_negative = compose(ImageRequest(
            prompt=prompt, provider=provider, model=model,
            negative_prompt=negative, mode=mode,
        ))
        with st.expander("Final prompts sent to the generator"):
            st.code(preview_prompt or "(enter a prompt)", language=None)
            if provider == "local":
                st.caption("Diffusion negative prompt")
                st.code(preview_negative or "(none)", language=None)
    
        if st.button("Generate images", type="primary", disabled=not prompt.strip(),
                     use_container_width=True):
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
                req = ImageRequest(
                    prompt=prompt, negative_prompt=negative, provider=provider, model=model,
                    mode=mode, width=width, height=height, steps=steps,
                    guidance=guidance, seed=int(seed), count=count,
                    input_image=store_upload(reference) if reference else None,
                    reference_images=tuple(store_upload(f) for f in extras),
                    strength=strength, quality=quality,
                )
                prompt_sources = {
                    "categories": {
                        category: {
                            "preset_groups": selected_components(category),
                            "tags": list(st.session_state.get(f"{category}_tags", [])),
                        }
                        for category in ("characters", "clothing", "poses", "backgrounds", "other")
                    },
                    "recipe": st.session_state.get("recipe_selection", "custom"),
                    "manual_prompt_override": bool(
                        st.session_state.get("_manual_prompt_override")
                    ),
                }
                overall_progress = st.progress(0.0, text="Preparing image generation…")
                status = st.empty()
                elapsed_display = st.empty()
                preview_columns = st.columns(min(count, 4))
                image_progress = [
                    column.progress(0.0, text=f"Image {index + 1}: waiting")
                    for index, column in enumerate(preview_columns)
                ]
                image_slots = [column.empty() for column in preview_columns]
                events = queue.Queue()
                completed = threading.Event()
                result = {}
    
                def update_progress(event):
                    if event.get("retrying"):
                        for index in range(event["batch_start"], count):
                            image_progress[index].progress(0.0, text=f"Image {index + 1}: retrying")
                        overall_progress.progress(event.get("fraction", 0.0),
                                                  text="Retrying with a smaller GPU batch…")
                        status.warning("GPU memory limit reached; retrying with a smaller batch.")
                        return
                    fraction = event.get("fraction", 0.0)
                    overall_progress.progress(fraction, text=f"Overall generation: {fraction:.0%}")
                    start = event.get("batch_start", 0)
                    end = min(count, start + event.get("batch_size", 1))
                    image_fraction = event.get("image_fraction", 0.0)
                    for index in range(start, end):
                        image_progress[index].progress(
                            image_fraction,
                            text=f"Image {index + 1}: {image_fraction:.0%}",
                        )
                    if event.get("step"):
                        if strategy == "sequential":
                            label = (f"Image {event['image_index']} of {count} · "
                                     f"step {event['step']} of {event['steps']}")
                        else:
                            label = (f"Batch of {count} images · "
                                     f"step {event['step']} of {event['steps']}")
                        status.info(label)
    
                def show_image(image, index, _total):
                    image_progress[index - 1].progress(1.0, text=f"Image {index}: complete")
                    image_slots[index - 1].image(image, use_container_width=True)
                    status.success(f"Image {index} of {count} finished.")
    
                def generation_worker():
                    try:
                        result["output_dir"] = run_generation(
                            req, strategy=strategy,
                            progress_callback=lambda event: events.put(("progress", event)),
                            image_callback=lambda image, index, total: events.put(
                                ("image", (image.copy(), index, total))
                            ),
                            stage_callback=lambda stage: events.put(("stage", stage)),
                            prompt_sources=prompt_sources,
                        )
                    except Exception as exc:
                        LOGGER.exception("Generation failed for model %s", req.model)
                        result["error"] = (str(exc), traceback.format_exc())
                    finally:
                        completed.set()
    
                LOGGER.info(
                    "Generation requested: provider=%s model=%s count=%d size=%dx%d steps=%d guidance=%.2f seed=%d",
                    provider, model, count, width, height, steps, guidance, int(seed),
                )
                started = time.perf_counter()
                next_clock_tick = started
                worker = threading.Thread(
                    target=generation_worker, name="suwako-generation", daemon=True
                )
                worker.start()
                elapsed_display.caption("Generation running · elapsed 0s")
                while not completed.is_set() or not events.empty():
                    try:
                        event_type, payload = events.get(timeout=0.2)
                    except queue.Empty:
                        event_type = payload = None
                    if event_type == "progress":
                        update_progress(payload)
                        if payload.get("step") and (
                            payload["step"] in (1, payload["steps"])
                            or payload["step"] % max(1, payload["steps"] // 4) == 0
                        ):
                            LOGGER.debug(
                                "Denoising progress: model=%s image=%d/%d step=%d/%d",
                                model, payload["image_index"], count,
                                payload["step"], payload["steps"],
                            )
                    elif event_type == "image":
                        show_image(*payload)
                        LOGGER.info("Image %d of %d completed", payload[1], count)
                    elif event_type == "stage":
                        LOGGER.info("Generation stage: %s", payload)
                        status.info(payload)
                    now = time.perf_counter()
                    if now >= next_clock_tick:
                        elapsed_display.caption(
                            f"Generation in progress · elapsed {format_duration(now - started)}"
                        )
                        next_clock_tick = now + 1.0
    
                if "error" in result:
                    message, details = result["error"]
                    elapsed = time.perf_counter() - started
                    status.error(f"Generation failed: {message}")
                    elapsed_display.caption(f"Failed after {format_duration(elapsed)}")
                    with st.expander("Debug traceback"):
                        st.code(details, language="text")
                else:
                    output_dir = result["output_dir"]
                    elapsed = time.perf_counter() - started
                    overall_progress.progress(1.0, text="Generation and saving complete")
                    elapsed_display.caption(f"Finished in {format_duration(elapsed)}")
                    LOGGER.info("Generation completed in %s; saved run %s",
                                format_duration(elapsed), output_dir)
                    st.session_state["last_run"] = str(output_dir)
                    st.success(f"Saved {count} image(s) to {output_dir}")
            except Exception as exc:
                LOGGER.exception("Generation UI error")
                st.exception(exc)

        st_latest_run_curation()

    with review_tab:
        st_gallery_fancy()

if __name__ == "__main__":
    main()
