# Changelog

Notable Suwako Diffusion changes are recorded here.

## Unreleased

- Add a searchable picture library with physical Inbox and album folders, tags, batch
  moves, multi-file imports, and paged thumbnail browsing.
- Store generation manifests separately from library images and link them with stable
  asset IDs; record library-selected img2img source assets in new runs.
- Add verified, rollback-safe migration from legacy character/run output folders.
- Start new sessions with Alexa's compact competition-pool setup and allow saving the current local configuration for next time.
- Trim the default negative prompt to reduce CLIP token truncation.
- Include selected preset context, model, seed, and run ID in future image filenames.
- Add Alexa competition swimsuit and indoor competition pool presets.
- Add a compact Alexa competition-pool recipe for pose testing.
- Add Alexa Mikazuchi character and rigging-swimsuit presets.
- Add compact hogtie, ball-tie, and carried-hogtie pose presets.
- Add Black Lotus grand lobby and VIP lounge environment presets, with Annette test recipes.
- Added a once-per-second elapsed timer that includes model loading, generation, and saving.
- Kept generation on a single serialized GPU worker while Streamlit polls progress events.
- Split prompt components into JSON files under `data/presets/` for characters, clothing, poses, backgrounds, and other tags.
- Migrated the legacy character, outfit, pose, background, style, and named example combinations into the preset catalog.
- Added tabbed prompt editing and an editable assembled prompt, with prompt source selections saved in run metadata.
- Moved generation controls into a collapsed advanced section with defaults of 28 steps, 5.5 guidance, seed 0, 1024×1024 resolution, and 0.65 img2img strength.
- Added per-process application logs with console output and exception tracebacks; desktop Streamlit output is relayed to the launcher console and a timestamped log file.
- Store generated runs in character folders with readable prompt-based run names.
- Add a dedicated review gallery with image downloads, per-image deletion, folder
  opening, character filtering, and support for legacy output locations.
- Add checkbox-based batch deletion that updates run metadata while keeping unselected
  images intact.
- Add a gallery cleanup action for run folders that contain metadata but no images.
- Add a default `masterpiece` suffix preset to the Other prompt group.
- Add per-image thumbs-up and thumbs-down review actions: favorites move into a
  character-specific `best` folder by default, with a gallery option to rename it;
  dislikes are deleted immediately.
- Add a latest-batch curation panel on the Generate tab, with the same actions as
  the saved Review gallery.
- Add a saved-preset editor that writes character, clothing, pose, and background tag
  changes back to their JSON files.
- Group reference-image inputs and img2img strength in a collapsed options section,
  with previews for uploaded reference images.
