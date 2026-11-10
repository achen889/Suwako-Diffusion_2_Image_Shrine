"""Compose prompts without silently changing exact-mode user input."""
import json
from pathlib import Path

from .models import ImageRequest, LOCAL_MODELS

PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "presets"
PRESET_CATEGORIES = ("characters", "clothing", "poses", "backgrounds", "other")


def load_prompt_config() -> dict:
    """Load categorized prompt components and saved recipes from JSON files."""
    defaults_path = PROMPT_CONFIG_PATH / "defaults.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    config = {"default_negative_prompt": defaults.get("default_negative_prompt", ""),
              "categories": {}, "recipes": []}
    for category in PRESET_CATEGORIES:
        entries = []
        for path in sorted((PROMPT_CONFIG_PATH / category).glob("*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            if not entry.get("key") or not entry.get("name"):
                raise ValueError(f"Preset {path} needs a key and name.")
            if not isinstance(entry.get("tags"), list):
                raise ValueError(f"Preset {path} needs a list of prompt tags.")
            entries.append(entry)
        config["categories"][category] = entries
    for path in sorted((PROMPT_CONFIG_PATH / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        if not recipe.get("key") or not recipe.get("name"):
            raise ValueError(f"Recipe {path} needs a key and name.")
        config["recipes"].append(recipe)
    return config


def tags_for_components(entries: list[dict], selected: list[str]) -> list[str]:
    """Flatten selected component files into ordered, unique prompt tags."""
    lookup = {entry["key"]: entry for entry in entries}
    tags = []
    seen = set()
    for key in selected:
        for tag in lookup.get(key, {}).get("tags", []):
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def assemble_prompt(selections: dict[str, list[str]], config: dict) -> str:
    """Assemble tags in the studio's stable character-to-style order."""
    categories = config["categories"]
    return join_prompt_tags([
        tag
        for category in PRESET_CATEGORIES
        for tag in tags_for_components(categories[category], selections.get(category, []))
    ])


def assemble_selected_tags(selected: dict[str, list[str]]) -> str:
    """Join already-edited tags in the same stable category order."""
    return join_prompt_tags([
        tag for category in PRESET_CATEGORIES for tag in selected.get(category, [])
    ])


def split_prompt_tags(value: str) -> list[str]:
    """Split comma-separated prompt tags while preserving their order."""
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def join_prompt_tags(tags: list[str]) -> str:
    """Join selected prompt tags into the editable prompt field format."""
    return ", ".join(tag.strip() for tag in tags if tag.strip())


def compose(request: ImageRequest) -> tuple[str, str]:
    prompt = request.prompt.strip()
    negative = request.negative_prompt.strip()
    if request.provider == "local" and request.mode == "enhanced":
        profile = LOCAL_MODELS[request.model]
        if profile.prefix:
            prompt = ", ".join((profile.prefix, prompt.lstrip(" ,")))
        if profile.suffix:
            prompt = ", ".join((prompt.rstrip(" ,"), profile.suffix))
        negative = ", ".join(p for p in (profile.negative, negative) if p)
    if request.provider == "openai" and negative:
        prompt += "\n\nAvoid these features: " + negative
    return prompt, negative
