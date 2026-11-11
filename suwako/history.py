"""Each run is portable: PNG files plus JSON settings and execution metadata."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
import re
import uuid

from .models import ImageRequest

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
GALLERY_SETTINGS_NAME = ".gallery-settings.json"
DEFAULT_BEST_FOLDER = "best"


def _safe_slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:64].rstrip("_") or fallback


def _character_folder(details: dict) -> str:
    sources = details.get("prompt_sources") or {}
    categories = sources.get("categories") or {}
    characters = (categories.get("characters") or {}).get("preset_groups") or []
    if isinstance(characters, str):
        characters = [characters]
    slugs = sorted({_safe_slug(str(key), "character") for key in characters})
    if not slugs:
        return "uncategorized"
    if len(slugs) == 1:
        return slugs[0]
    return "multiple_characters"


def save_run(request: ImageRequest, images, details: dict, elapsed: float,
             root: Path = OUTPUT_ROOT) -> Path:
    now = datetime.now(timezone.utc)
    prompt_slug = _safe_slug(request.prompt, "image")
    run_name = (now.strftime("%Y%m%dT%H%M%SZ") + "_" + prompt_slug[:48].rstrip("_")
                + "_" + uuid.uuid4().hex[:8])
    directory = root / _character_folder(details) / run_name
    directory.mkdir(parents=True, exist_ok=False)
    names = []
    for index, img in enumerate(images, 1):
        name = f"image_{index:02d}.png"
        img.save(directory / name, format="PNG")
        names.append(name)
    data = asdict(request)
    data["input_image"] = str(request.input_image) if request.input_image else None
    data["reference_images"] = [str(p) for p in request.reference_images]
    data.update(schema_version=3, created_at=now.isoformat(),
                elapsed_seconds=round(elapsed, 3), images=names, **details)
    temp = directory / "metadata.json.tmp"
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(directory / "metadata.json")
    return directory


def recent_runs(root: Path = OUTPUT_ROOT, limit: int = 30):
    if not root.is_dir():
        return []
    results = []
    metadata_files = sorted(
        root.rglob("metadata.json"),
        key=lambda path: (path.parent.name, str(path.parent)),
        reverse=True,
    )
    for meta in metadata_files:
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            if isinstance(data.get("images"), list):
                results.append((meta.parent, data))
        except (OSError, ValueError, TypeError):
            continue
        if len(results) >= limit:
            break
    return results


def empty_run_folders(root: Path = OUTPUT_ROOT) -> list[Path]:
    """Find output run folders containing valid metadata and no other files."""
    root = root.resolve()
    if not root.is_dir():
        return []
    empty = []
    for metadata_file in root.rglob("metadata.json"):
        directory = metadata_file.parent
        if (not directory.is_dir() or directory.is_symlink()
                or directory.resolve() == root
                or not directory.resolve().is_relative_to(root)):
            continue
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
            entries = list(directory.iterdir())
        except (OSError, ValueError, TypeError):
            continue
        if (isinstance(data, dict)
                and isinstance(data.get("images"), list)
                and len(entries) == 1
                and entries[0].name == "metadata.json"
                and entries[0].is_file()
                and not entries[0].is_symlink()):
            empty.append(directory)
    return empty


def cleanup_empty_run_folders(root: Path = OUTPUT_ROOT) -> list[Path]:
    """Remove only run folders that contain a valid metadata.json and nothing else."""
    root = root.resolve()
    removed = []
    for directory in empty_run_folders(root):
        metadata_file = directory / "metadata.json"
        if (not directory.resolve().is_relative_to(root)
                or directory.is_symlink()
                or not metadata_file.is_file()
                or metadata_file.is_symlink()):
            continue
        entries = list(directory.iterdir())
        if len(entries) != 1 or entries[0] != metadata_file:
            continue
        metadata_file.unlink()
        directory.rmdir()
        removed.append(directory)
    return removed


def _write_metadata(directory: Path, data: dict) -> None:
    metadata_file = directory / "metadata.json"
    temp = metadata_file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(metadata_file)


def _validate_collection_name(name: str) -> str:
    value = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}", value):
        raise ValueError("Folder name must use 1–40 letters, numbers, spaces, '_' or '-'.")
    if value.split(" ", 1)[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError("That name is reserved by Windows.")
    return value


def best_folder_name(root: Path = OUTPUT_ROOT) -> str:
    """Return the configured name for images marked as best, defaulting to best."""
    settings = root / GALLERY_SETTINGS_NAME
    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
        return _validate_collection_name(data.get("best_folder", DEFAULT_BEST_FOLDER))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return DEFAULT_BEST_FOLDER


def rename_best_folder(name: str, root: Path = OUTPUT_ROOT) -> str:
    """Rename each character's best collection and persist the selected folder name."""
    name = _validate_collection_name(name)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    old_name = best_folder_name(root)
    if name.casefold() == old_name.casefold():
        return old_name

    plans = []
    for character_dir in root.iterdir():
        if (not character_dir.is_dir() or character_dir.name.startswith(".")
                or (character_dir / "metadata.json").is_file()):
            continue
        source = character_dir / old_name
        destination = character_dir / name
        if not source.is_dir():
            continue
        if not source.resolve().is_relative_to(root):
            raise ValueError("Collection folder resolves outside the outputs folder.")
        if destination.exists():
            if not destination.is_dir() or not destination.resolve().is_relative_to(root):
                raise ValueError("The new collection path is not a safe folder.")
            collisions = [item.name for item in source.iterdir()
                          if (destination / item.name).exists()]
            if collisions:
                raise FileExistsError(
                    f"Cannot merge folders; these run names already exist: {', '.join(collisions)}"
                )
            plans.append((source, destination, True))
        else:
            plans.append((source, destination, False))

    moved = []
    created_destinations = []
    try:
        for source, destination, merge in plans:
            if merge:
                destination.mkdir(parents=True, exist_ok=True)
                created_destinations.append(destination)
                for item in list(source.iterdir()):
                    moved_item = destination / item.name
                    item.rename(moved_item)
                    moved.append((item, moved_item))
                source.rmdir()
            else:
                source.rename(destination)
                moved.append((source, destination))

        settings_path = root / GALLERY_SETTINGS_NAME
        temp = settings_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps({"best_folder": name}, indent=2) + "\n",
                          encoding="utf-8")
        temp.replace(settings_path)
    except OSError:
        for original, current in reversed(moved):
            if current.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                current.rename(original)
        for destination in reversed(created_destinations):
            try:
                destination.rmdir()
            except OSError:
                pass
        raise
    return name


def _listed_image(directory: Path, name: str, root: Path) -> tuple[Path, dict]:
    root = root.resolve()
    directory = directory.resolve()
    if not directory.is_relative_to(root) or Path(name).name != name:
        raise ValueError("Image must be a listed file inside the outputs folder.")
    file = directory / name
    if not file.is_file() or not file.resolve().is_relative_to(root):
        raise ValueError("Image file is missing or outside the outputs folder.")
    metadata_file = directory / "metadata.json"
    if not metadata_file.resolve().is_relative_to(root):
        raise ValueError("Run metadata points outside the outputs folder.")
    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    if name not in data.get("images", []):
        raise ValueError("Image is not listed in this run's metadata.")
    return file, data


def keep_best(directory: Path, name: str, root: Path = OUTPUT_ROOT) -> Path:
    """Move one saved image into its character's best folder, retaining provenance."""
    file, data = _listed_image(directory, name, root)
    directory = directory.resolve()
    root = root.resolve()
    if data.get("best_of") or directory.parent.name.casefold() in {
        DEFAULT_BEST_FOLDER.casefold(), best_folder_name(root).casefold(),
    }:
        raise ValueError("Image is already in the best folder.")

    character = directory.parent.name
    destination_dir = root / character / best_folder_name(root) / directory.name
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not destination_dir.resolve().is_relative_to(root):
        raise ValueError("Best folder resolves outside the outputs folder.")
    destination = destination_dir / name
    if destination.exists():
        raise FileExistsError(f"Best image already exists: {destination}")

    # Copy first so a metadata write failure never loses the original image.
    shutil.copy2(file, destination)
    best_data = dict(data)
    best_data["images"] = [name]
    best_data["best_of"] = {"run": directory.name, "image": name}
    _write_metadata(destination_dir, best_data)

    data["images"] = [item for item in data.get("images", []) if item != name]
    data.setdefault("ratings", {})[name] = "best"
    _write_metadata(directory, data)
    file.unlink()
    return destination


def discard_image(directory: Path, name: str, root: Path = OUTPUT_ROOT) -> None:
    """Remove a reviewed image and its listing while preserving the rest of the run."""
    file, data = _listed_image(directory, name, root)
    data["images"] = [item for item in data.get("images", []) if item != name]
    data.setdefault("ratings", {})[name] = "discarded"
    _write_metadata(directory.resolve(), data)
    file.unlink()


def discard_images(images: list[tuple[Path, str]], root: Path = OUTPUT_ROOT) -> int:
    """Delete a batch of listed images while preserving all other run files."""
    unique = []
    seen = set()
    for directory, name in images:
        key = (directory.resolve(), name)
        if key not in seen:
            seen.add(key)
            unique.append((directory, name))
    if not unique:
        return 0

    validated = []
    grouped: dict[Path, tuple[dict, list[tuple[str, Path]]]] = {}
    for directory, name in unique:
        file, data = _listed_image(directory, name, root)
        resolved_dir = directory.resolve()
        validated.append((resolved_dir, name, file))
        if resolved_dir not in grouped:
            grouped[resolved_dir] = (data, [])
        grouped[resolved_dir][1].append((name, file))

    originals = {directory: dict(data) for directory, (data, _) in grouped.items()}
    try:
        for directory, (data, selected) in grouped.items():
            updated = dict(data)
            removed = {name for name, _ in selected}
            updated["images"] = [name for name in data.get("images", [])
                                 if name not in removed]
            ratings = dict(data.get("ratings") or {})
            ratings.update({name: "discarded" for name in removed})
            updated["ratings"] = ratings
            _write_metadata(directory, updated)
        for _, _, file in validated:
            file.unlink()
    except OSError:
        for directory, data in originals.items():
            _write_metadata(directory, data)
        raise
    return len(validated)
