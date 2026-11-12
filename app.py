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

from PIL import Image, ImageOps
import streamlit as st

from suwako.history import (
    OUTPUT_ROOT,
    best_folder_name,
    cleanup_empty_run_folders,
    discard_image,
    discard_images,
    empty_run_folders,
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
from suwako.settings import load_last_configuration, save_last_configuration
from suwako.library import Library
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
        selected = list(recipe.get("components", {}).get(category, []))
        if category == "other" and "masterpiece_suffix" in {
            item["key"] for item in config["categories"]["other"]
        }:
            selected = ["masterpiece_suffix", *[
                key for key in selected if key != "masterpiece_suffix"
            ]]
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
    saved = load_last_configuration()
    recipes = {item["key"]: item for item in config["recipes"]}
    if saved:
        categories = saved.get("categories", {})
        if not isinstance(categories, dict):
            categories = {}
        for category in config["categories"]:
            saved_category = categories.get(category, {})
            valid_keys = {item["key"] for item in config["categories"][category]}
            saved_groups = (
                saved_category.get("preset_groups", [])
                if isinstance(saved_category, dict) else []
            )
            if isinstance(saved_groups, str):
                saved_groups = [saved_groups]
            if not isinstance(saved_groups, list):
                saved_groups = []
            selected = [
                key for key in saved_groups
                if isinstance(key, str) and key in valid_keys
            ]
            tags = saved_category.get("tags", []) if isinstance(saved_category, dict) else []
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                tags = tags_for_components(config["categories"][category], selected)
            st.session_state[f"{category}_components"] = selected
            st.session_state[f"{category}_tags"] = tags
        saved_recipe = saved.get("recipe_selection")
        st.session_state["recipe_selection"] = (
            saved_recipe if isinstance(saved_recipe, str) and saved_recipe in recipes
            else "custom"
        )
        st.session_state["prompt_input"] = str(saved.get("prompt_input", ""))
        st.session_state["_manual_prompt_override"] = bool(
            saved.get("manual_prompt_override", False)
        )
        negative_prompt = saved.get("negative_prompt_input")
        st.session_state["negative_prompt_input"] = (
            negative_prompt if isinstance(negative_prompt, str)
            else config["default_negative_prompt"]
        )
    else:
        default_recipe = "alexa_competition_pool"
        if default_recipe in recipes:
            st.session_state["recipe_selection"] = default_recipe
            apply_recipe(config)
        else:
            initial = {
                "characters": ["alexa_mikazuchi"],
                "clothing": ["alexa_competition_swimsuit"],
                "poses": [],
                "backgrounds": ["indoor_competition_pool"],
                "other": ["masterpiece_suffix"],
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
            st.session_state["recipe_selection"] = "custom"
        st.session_state["_manual_prompt_override"] = False
        st.session_state["negative_prompt_input"] = config["default_negative_prompt"]
    st.session_state["negative_prompt_tags"] = split_prompt_tags(
        st.session_state["negative_prompt_input"]
    )
    if saved:
        widget_defaults = {
            "provider_selection": ("local", ("local", "openai")),
            "prompt_mode": ("enhanced", ("enhanced", "exact")),
            "image_count": (1, None),
            "resolution-local": ("1024x1024", ("512x512", "768x768", "1024x1024", "1024x1536", "1536x1024")),
            "resolution-openai": ("1024x1024", ("1024x1024", "1536x1024", "1024x1536")),
            "sampling_steps": (28, None),
            "guidance_scale": (5.5, None),
            "seed_value": (0, None),
            "img2img_strength": (0.65, None),
            "generation_strategy": ("Batched", ("Batched", "One at a time (live previews)")),
            "openai_quality": ("medium", ("low", "medium", "high")),
        }
        saved_fields = {
            "provider_selection": "provider",
            "prompt_mode": "mode",
            "image_count": "count",
            "resolution-local": "resolution_local",
            "resolution-openai": "resolution_openai",
            "sampling_steps": "steps",
            "guidance_scale": "guidance",
            "seed_value": "seed",
            "img2img_strength": "strength",
            "generation_strategy": "strategy_label",
            "openai_quality": "quality",
        }
        for key, (default, choices) in widget_defaults.items():
            value = saved.get(saved_fields[key], default)
            if choices is not None and value not in choices:
                value = default
            if key in ("image_count", "sampling_steps", "seed_value"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = default
                lower, upper = {
                    "image_count": (1, 4),
                    "sampling_steps": (5, 60),
                    "seed_value": (0, 2**31 - 1),
                }[key]
                if not lower <= value <= upper:
                    value = default
            elif key in ("guidance_scale", "img2img_strength"):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = default
                lower, upper = (0.0, 15.0) if key == "guidance_scale" else (0.05, 1.0)
                if not lower <= value <= upper:
                    value = default
            st.session_state[key] = value
        provider = st.session_state["provider_selection"]
        saved_model = saved.get("model")
        available_models = LOCAL_MODELS if provider == "local" else OPENAI_MODELS
        st.session_state["model_selection"] = (
            saved_model if isinstance(saved_model, str) and saved_model in available_models
            else next(iter(available_models))
        )
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


def st_legacy_gallery_fancy():
    """Render saved images as review cards with explicit local file actions."""
    output_root = OUTPUT_ROOT.resolve()

    with st.expander("Move legacy generations into the picture library", expanded=False):
        st.caption(
            "Moves listed images from the old character/run folders into Inbox and the "
            "current Best album. Run settings and provenance are retained."
        )
        confirm_migration = st.checkbox(
            "Confirm moving legacy image files and run manifests",
            key="gallery-confirm-library-migration",
        )
        if st.button(
            "Migrate legacy generations",
            key="gallery-migrate-legacy-generations",
            disabled=not confirm_migration,
        ):
            try:
                result = Library(OUTPUT_ROOT).migrate_legacy_outputs(
                    best_folder=best_folder_name(OUTPUT_ROOT)
                )
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Migration stopped and rolled back: {exc}")
            else:
                st.success(
                    f"Moved {result['migrated_images']} image(s) from "
                    f"{result['migrated_runs']} run(s). Skipped "
                    f"{result['skipped_metadata']} unreadable metadata file(s)."
                )
                st.rerun()

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
    empty_folders = empty_run_folders(OUTPUT_ROOT)
    with st.expander(f"Clean up empty run folders ({len(empty_folders)})", expanded=False):
        if empty_folders:
            st.caption("These folders contain valid run metadata and no images or other files.")
            for directory in empty_folders[:30]:
                st.code(str(directory.relative_to(output_root)), language=None)
            if len(empty_folders) > 30:
                st.caption(f"And {len(empty_folders) - 30} more metadata-only folders.")
            if st.button(
                f"Delete {len(empty_folders)} empty run folders",
                key="gallery-clean-empty-runs",
                help="Remove only the listed run folders, each containing metadata.json and nothing else.",
            ):
                try:
                    removed = cleanup_empty_run_folders(OUTPUT_ROOT)
                except OSError as exc:
                    st.error(f"Could not clean up empty run folders: {exc}")
                else:
                    st.success(f"Removed {len(removed)} empty run folder(s).")
                    st.rerun()
        else:
            st.caption("No metadata-only run folders to remove.")

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
                    st.image(str(file), width="stretch")
                    st.checkbox("Select for deletion", key=selection_key(directory, name))
                    st.caption(f"{folder_name(directory, data)} · {directory.name}")
                    st.caption(f"{data.get('provider', '?')} / {data.get('model', '?')}")
                    best_dir = is_best_directory(directory, data)
                    rating_columns = st.columns(2)
                    with rating_columns[0]:
                        if st.button(
                            "👍 Best" if best_dir else f"👍 Keep {current_best_folder}",
                            key=f"gallery-best-{directory.name}-{name}",
                            disabled=best_dir, width="stretch",
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
                            width="stretch",
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
                        width="stretch",
                    )
                    if st.button(
                        "Open folder", key=f"gallery-open-{directory.name}-{name}",
                        width="stretch",
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


def st_gallery_fancy():
    """Browse and organize images in the managed local picture library."""
    library = Library(OUTPUT_ROOT)
    with st.expander("Move legacy generations into the picture library", expanded=False):
        st.caption("Moves old run images into Inbox and the current Best album, preserving run provenance.")
        confirmed = st.checkbox("Confirm moving legacy images and manifests",
                                key="library-migration-confirm")
        if st.button("Migrate legacy generations", key="library-migration-run",
                     disabled=not confirmed):
            try:
                result = library.migrate_legacy_outputs(
                    best_folder=best_folder_name(OUTPUT_ROOT))
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Migration stopped and rolled back: {exc}")
            else:
                st.success(f"Moved {result['migrated_images']} image(s) from "
                           f"{result['migrated_runs']} run(s); skipped "
                           f"{result['skipped_metadata']} unreadable manifest(s).")
                st.rerun()

    with st.expander("Import pictures", expanded=False):
        uploads = st.file_uploader("Copy pictures into Inbox",
                                   type=("png", "jpg", "jpeg", "webp"),
                                   accept_multiple_files=True, key="library-import-files") or []
        if st.button("Import selected files", key="library-import-button", disabled=not uploads):
            imported = 0
            try:
                for upload in uploads:
                    library.import_image(upload)
                    imported += 1
            except (OSError, ValueError) as exc:
                st.error(f"Imported {imported} file(s) before stopping: {exc}")
            else:
                st.success(f"Imported {imported} picture(s) into Inbox.")
                st.rerun()

    query = st.text_input("Search pictures", key="library-search")
    album_options = ["All pictures", "Inbox", *library.albums()]
    selected_album = st.selectbox("Folder", album_options, key="library-album-filter")
    selected_tags = st.multiselect("Tags", library.all_tags(), key="library-tag-filter")
    album_filter = None if selected_album == "All pictures" else selected_album
    filter_state = (query, album_filter, tuple(selected_tags))
    if st.session_state.get("_library-filter-state") != filter_state:
        st.session_state["_library-filter-state"] = filter_state
        st.session_state["library-page"] = 1

    total = library.count_assets(query=query, album=album_filter, tags=selected_tags)
    page_size = 48
    pages = max(1, (total + page_size - 1) // page_size)
    if int(st.session_state.get("library-page", 1)) > pages:
        st.session_state["library-page"] = pages
    page = st.number_input("Page", min_value=1, max_value=pages,
                           value=min(int(st.session_state.get("library-page", 1)), pages),
                           step=1, key="library-page")
    assets = library.list_assets(query=query, album=album_filter, tags=selected_tags,
                                 offset=(int(page) - 1) * page_size, limit=page_size)
    st.caption(f"{total} picture(s) · page {page} of {pages} · {OUTPUT_ROOT}")

    if not assets:
        st.info("No pictures match this view. Import files or generate images into Inbox.")
    else:
        selected_ids = []
        columns = st.columns(4)
        for index, asset in enumerate(assets):
            with columns[index % 4]:
                with st.container(border=True):
                    st.image(str(library.thumbnail(asset.asset_id)), width="stretch")
                    if st.checkbox("Select", key=f"library-select-{asset.asset_id}"):
                        selected_ids.append(asset.asset_id)
                    st.caption(asset.display_name)
                    st.caption(f"{asset.model or asset.origin} · {asset.width}×{asset.height}")
                    if asset.prompt:
                        st.caption(asset.prompt[:160])
                    if asset.tags:
                        st.caption("Tags: " + ", ".join(asset.tags))
                    if st.button("Use as img2img input", key=f"library-use-input-{asset.asset_id}",
                                 width="stretch"):
                        st.session_state["_library_input_asset_id"] = asset.asset_id
                        st.toast("Selected as the next img2img input. Open Generate to continue.")
                    if st.button("Prepare download", key=f"library-prepare-download-{asset.asset_id}",
                                 width="stretch"):
                        st.session_state["_library-download-asset"] = asset.asset_id
                        st.rerun()
                    if st.session_state.get("_library-download-asset") == asset.asset_id:
                        st.download_button("Download image", asset.path.read_bytes(),
                                           file_name=asset.display_name,
                                           key=f"library-download-{asset.asset_id}",
                                           width="stretch")
                    if st.button("Delete image", key=f"library-delete-{asset.asset_id}",
                                 width="stretch"):
                        try:
                            library.remove_assets([asset.asset_id])
                        except (OSError, KeyError) as exc:
                            st.error(f"Could not delete image: {exc}")
                        else:
                            st.session_state.pop(f"library-select-{asset.asset_id}", None)
                            st.rerun()

        with st.expander(f"Organize selected pictures ({len(selected_ids)})", expanded=False):
            targets = ["Inbox", *library.albums()]
            target = st.selectbox("Move to folder", targets, key="library-move-target")
            tags_text = st.text_input("Tags (comma separated)", key="library-add-tags")
            tag_mode = st.radio("Tag update", ("Add to existing", "Replace existing"),
                                horizontal=True, key="library-tag-mode")
            if st.button("Apply to selected", key="library-organize-apply",
                         disabled=not selected_ids):
                try:
                    library.move_assets(selected_ids, target)
                    tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
                    if tag_mode == "Replace existing":
                        library.set_tags(selected_ids, tags)
                    elif tags:
                        library.add_tags(selected_ids, tags)
                except (OSError, ValueError, KeyError) as exc:
                    st.error(f"Could not organize selected pictures: {exc}")
                else:
                    st.success(f"Updated {len(selected_ids)} picture(s).")
                    st.rerun()
            confirmed_delete = st.checkbox(
                f"Confirm deleting {len(selected_ids)} selected picture(s)",
                key="library-delete-confirm", disabled=not selected_ids)
            if st.button("Delete selected pictures", key="library-delete-selected",
                         disabled=not selected_ids or not confirmed_delete):
                try:
                    deleted = library.remove_assets(selected_ids)
                except (OSError, KeyError) as exc:
                    st.error(f"Could not delete selected pictures: {exc}")
                else:
                    st.success(f"Deleted {deleted} picture(s).")
                    st.rerun()

    with st.expander("Manage albums", expanded=False):
        create_name = st.text_input("New album name", key="library-new-album")
        if st.button("Create album", key="library-create-album"):
            try:
                library.create_album(create_name)
            except (OSError, ValueError) as exc:
                st.error(f"Could not create album: {exc}")
            else:
                st.rerun()
        albums = library.albums()
        if albums:
            old_name = st.selectbox("Album to rename", albums, key="library-rename-source")
            new_name = st.text_input("New album name", key="library-rename-target")
            if st.button("Rename album", key="library-rename-album"):
                try:
                    if old_name.casefold() == best_folder_name(OUTPUT_ROOT).casefold():
                        rename_best_folder(new_name, OUTPUT_ROOT)
                    else:
                        library.rename_album(old_name, new_name)
                except (OSError, ValueError) as exc:
                    st.error(f"Could not rename album: {exc}")
                else:
                    st.rerun()

    legacy = [(directory, data) for directory, data in recent_runs(OUTPUT_ROOT, limit=200)
              if int(data.get("schema_version", 0) or 0) < 4]
    if legacy:
        count = sum(len(data.get("images", [])) for _, data in legacy)
        st.info(f"{count} legacy image(s) remain outside the library. Migrate them above to organize them here.")


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
    library = Library(OUTPUT_ROOT)
    assets = []
    asset_ids = data.get("asset_ids")
    if isinstance(asset_ids, list):
        for index, asset_id in enumerate(asset_ids):
            try:
                asset = library.get_asset(asset_id)
            except KeyError:
                continue
            if asset.path.is_file() and asset.path.resolve().is_relative_to(output_root):
                name = data.get("images", [])[index] if index < len(data.get("images", [])) else asset.display_name
                assets.append((name, asset_id, asset.path))
    else:
        for name in data.get("images", []):
            if (isinstance(name, str) and Path(name).name == name
                    and (directory / name).is_file()
                    and (directory / name).resolve().is_relative_to(output_root)):
                assets.append((name, None, directory / name))
    if not assets:
        return
    images = [entry[0] for entry in assets]
    with st.expander("Curate latest batch", expanded=False):
        batch_message = st.session_state.pop("_latest_batch_message", None)
        if batch_message:
            st.warning(batch_message)
        st.caption(f"{len(images)} image(s) · previews are shown at a fixed 360 × 360 px.")
        batch_actions = st.columns(2)
        with batch_actions[0]:
            if st.button(
                f"👍 Keep all {len(images)} as best",
                key=f"latest-best-all-{directory.name}",
                width="stretch",
                help="Move every remaining image in this batch to the best folder.",
            ):
                moved = 0
                try:
                    if asset_ids:
                        best_name = best_folder_name(OUTPUT_ROOT)
                        if best_name not in library.albums():
                            library.create_album(best_name)
                        library.move_assets([entry[1] for entry in assets], best_name)
                        moved = len(assets)
                    else:
                        for name in images:
                            keep_best(directory, name, OUTPUT_ROOT)
                            moved += 1
                except (OSError, ValueError) as exc:
                    st.session_state["_latest_batch_message"] = (
                        f"Moved {moved} image(s) before stopping: {exc}"
                    )
                    st.rerun()
                else:
                    st.success(f"Moved {moved} image(s) to the best folder.")
                    st.rerun()
        with batch_actions[1]:
            confirm_delete = st.checkbox(
                f"Confirm deleting all {len(images)} images",
                key=f"latest-confirm-delete-{directory.name}",
            )
            if st.button(
                f"👎 Delete all {len(images)}",
                key=f"latest-delete-all-{directory.name}",
                disabled=not confirm_delete,
                width="stretch",
                help="Permanently delete all remaining images in this batch.",
            ):
                try:
                    deleted = (library.remove_assets([entry[1] for entry in assets])
                               if asset_ids else discard_images(
                                   [(directory, name) for name in images], OUTPUT_ROOT))
                except (OSError, ValueError) as exc:
                    st.error(f"Could not delete the batch: {exc}")
                else:
                    st.success(f"Deleted {deleted} image(s).")
                    st.rerun()

        columns = st.columns(min(4, len(images)))
        for column, (name, asset_id, file) in zip(columns, assets):
            with column:
                if asset_id:
                    st.image(str(library.thumbnail(asset_id)), width=360)
                else:
                    with Image.open(file) as source:
                        preview = Image.new("RGB", (360, 360), "#161616")
                        fitted = ImageOps.contain(source.convert("RGB"), preview.size)
                        preview.paste(fitted, ((360 - fitted.width) // 2,
                                               (360 - fitted.height) // 2))
                    st.image(preview, width=360)
                actions = st.columns(2)
                with actions[0]:
                    if st.button(
                                 f"👍 Keep {best_folder_name(OUTPUT_ROOT)}",
                                 key=f"latest-best-{directory.name}-{name}",
                                 width="stretch"):
                        try:
                            if asset_id:
                                best_name = best_folder_name(OUTPUT_ROOT)
                                if best_name not in library.albums():
                                    library.create_album(best_name)
                                library.move_assets([asset_id], best_name)
                            else:
                                keep_best(directory, name, OUTPUT_ROOT)
                        except (OSError, ValueError) as exc:
                            st.error(f"Could not move image to best: {exc}")
                        else:
                            st.rerun()
                with actions[1]:
                    if st.button("👎 Delete", key=f"latest-delete-{directory.name}-{name}",
                                 help="Delete this image from disk immediately.",
                                 width="stretch"):
                        try:
                            if asset_id:
                                library.remove_assets([asset_id])
                            else:
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
                            "Local CUDA + Hugging Face" if x == "local" else "OpenAI Images API",
                            key="provider_selection")
        options = LOCAL_MODELS if provider == "local" else OPENAI_MODELS
        if st.session_state.get("model_selection") not in options:
            st.session_state["model_selection"] = next(iter(options))
        model = st.selectbox("Model", list(options), format_func=lambda key:
                             LOCAL_MODELS[key].name if provider == "local" else key,
                             key="model_selection")
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
                        "Load a recipe",
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
        mode = st.radio("Prompt processing", ("enhanced", "exact"), horizontal=True,
                        key="prompt_mode")
    
        with st.expander("Image-to-image options", expanded=False):
            reference = st.file_uploader(
                "Input image (img2img for local / image edit for OpenAI)",
                type=("png", "jpg", "jpeg", "webp"),
                key="input_reference_image",
            )
            library_input_id = st.session_state.get("_library_input_asset_id")
            library_input = None
            if library_input_id:
                try:
                    library_input = Library(OUTPUT_ROOT).get_asset(library_input_id)
                    if not library_input.path.is_file():
                        library_input = None
                except KeyError:
                    st.session_state.pop("_library_input_asset_id", None)
                    library_input_id = None
            if library_input is not None:
                st.image(str(library_input.path),
                         caption=f"Library input: {library_input.display_name}",
                         width="stretch")
                st.caption("The selected library picture takes priority over an uploaded input image.")
                if st.button("Clear selected library input", key="clear-library-input"):
                    st.session_state.pop("_library_input_asset_id", None)
                    st.rerun()
            if reference is not None:
                st.image(
                    reference.getvalue(), caption=f"Reference image: {reference.name}",
                    width="stretch",
                )
            has_input = reference is not None or library_input is not None
            if provider == "local":
                strength = st.slider(
                    "Img2img strength", 0.05, 1.0, 0.65, 0.05,
                    disabled=not has_input,
                    key="img2img_strength",
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
                    st.image(extra.getvalue(), caption=extra.name, width="stretch")
    
        count = st.slider("Images per request", 1, 4, 4 if provider == "local" else 1,
                          key="image_count")
        with st.expander("Advanced generation settings", expanded=False):
            if provider == "local":
                size_opts = ("512x512", "768x768", "1024x1024", "1024x1536", "1536x1024")
                if st.session_state.get("resolution-local") not in size_opts:
                    st.session_state["resolution-local"] = "1024x1024"
                size = st.selectbox("Resolution", size_opts, index=2, key="resolution-local")
                steps = st.slider("Sampling steps", 5, 60, 28, key="sampling_steps")
                guidance = st.slider("CFG / Guidance", 0.0, 15.0, 5.5, 0.5,
                                     key="guidance_scale")
                seed = st.number_input("Seed (0 = random)", 0, 2**31 - 1, 0,
                                       key="seed_value")
                strategy_label = st.radio(
                    "GPU execution", ("Batched", "One at a time (live previews)"),
                    help="Both modes use one GPU pipeline at a time. Batched mode is usually faster; one-at-a-time mode displays each image as soon as it finishes.",
                    horizontal=True,
                    key="generation_strategy",
                )
                strategy = "sequential" if strategy_label.startswith("One at a time") else "batch"
                st.caption("CUDA work stays serialized. Batched mode halves the batch automatically after an out-of-memory error.")
                quality = "medium"
            else:
                openai_sizes = ("1024x1024", "1536x1024", "1024x1536")
                if st.session_state.get("resolution-openai") not in openai_sizes:
                    st.session_state["resolution-openai"] = "1024x1024"
                size = st.selectbox("Resolution", openai_sizes, key="resolution-openai")
                quality = st.selectbox("API quality", ("low", "medium", "high"),
                                       index=1, key="openai_quality")
                steps, guidance, seed = 28, 5.0, 0
                strategy = "batch"
                st.caption("API has no diffusion sampler, CFG or reproducible seed controls.")
        width, height = map(int, size.split("x"))
        input_image_path = (
            library_input.path if library_input is not None else
            store_upload(reference) if reference is not None else None
        )

        if st.button("Save this setup for next time", key="save-last-configuration"):
            saved_configuration = {
                "schema_version": 1,
                "categories": {
                    category: {
                        "preset_groups": selected_components(category),
                        "tags": list(st.session_state.get(f"{category}_tags", [])),
                    }
                    for category in ("characters", "clothing", "poses", "backgrounds", "other")
                },
                "prompt_input": prompt,
                "manual_prompt_override": bool(
                    st.session_state.get("_manual_prompt_override")
                ),
                "negative_prompt_input": negative,
                "recipe_selection": st.session_state.get("recipe_selection", "custom"),
                "provider": provider,
                "model": model,
                "mode": mode,
                "count": count,
                "resolution_local": size if provider == "local" else
                                    st.session_state.get("resolution-local", "1024x1024"),
                "resolution_openai": size if provider == "openai" else
                                     st.session_state.get("resolution-openai", "1024x1024"),
                "steps": steps,
                "guidance": guidance,
                "seed": int(seed),
                "strength": strength,
                "strategy_label": strategy_label if provider == "local" else "Batched",
                "quality": quality,
            }
            try:
                save_last_configuration(saved_configuration)
            except OSError as exc:
                st.error(f"Could not save this configuration: {exc}")
            else:
                st.success("Saved locally. This setup will load next time the studio opens.")
        st.caption("The saved setup includes prompt selections and generation settings, not uploaded reference images.")
    
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
                     width="stretch"):
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
                req = ImageRequest(
                    prompt=prompt, negative_prompt=negative, provider=provider, model=model,
                    mode=mode, width=width, height=height, steps=steps,
                    guidance=guidance, seed=int(seed), count=count,
                    input_image=input_image_path,
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
                    image_slots[index - 1].image(image, width="stretch")
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
                            input_asset_id=(library_input_id if library_input is not None else None),
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
