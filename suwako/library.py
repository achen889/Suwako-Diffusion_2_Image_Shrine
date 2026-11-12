"""Local image library with physical albums and a rebuildable file catalog."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_PAGE_SIZE = 48


@dataclass(frozen=True)
class LibraryAsset:
    asset_id: str
    path: Path
    origin: str
    run_id: str | None
    created_at: str
    display_name: str
    prompt: str
    model: str
    tags: tuple[str, ...]
    width: int
    height: int


class Library:
    """Manage owned image files, albums, tags, and their searchable catalog."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root).resolve()
        self.library_root = self.output_root / "library"
        self.inbox = self.library_root / "Inbox"
        self.albums_root = self.library_root / "Albums"
        self.thumbnail_root = self.library_root / ".thumbnails"
        self.database = self.library_root / "catalog.sqlite3"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.albums_root.mkdir(parents=True, exist_ok=True)
        self.thumbnail_root.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._recover_migration()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    origin TEXT NOT NULL,
                    run_id TEXT,
                    created_at TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    checksum TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tags (
                    tag TEXT PRIMARY KEY COLLATE NOCASE
                );
                CREATE TABLE IF NOT EXISTS asset_tags (
                    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
                    tag TEXT NOT NULL REFERENCES tags(tag) ON DELETE CASCADE,
                    PRIMARY KEY (asset_id, tag)
                );
                CREATE INDEX IF NOT EXISTS assets_created_idx ON assets(created_at DESC);
                CREATE INDEX IF NOT EXISTS assets_run_idx ON assets(run_id);
            """)

    def _asset_path(self, relative_path: str) -> Path:
        path = (self.output_root / relative_path).resolve()
        if not path.is_relative_to(self.library_root.resolve()):
            raise ValueError("Catalog path points outside the managed image library.")
        return path

    @staticmethod
    def _safe_album_name(name: str) -> str:
        value = name.strip()
        if (not value or value.casefold() == "inbox"
                or value in {".", ".."}
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}", value)):
            raise ValueError("Album names must use 1–40 letters, numbers, spaces, '_' or '-'.")
        if value.split(" ", 1)[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }:
            raise ValueError("That name is reserved by Windows.")
        return value

    @staticmethod
    def _tag_list(tags) -> list[str]:
        result = []
        seen = set()
        for raw in tags or ():
            tag = str(raw).strip()
            if not tag or len(tag) > 80:
                continue
            key = tag.casefold()
            if key not in seen:
                seen.add(key)
                result.append(tag)
        return result

    def albums(self) -> list[str]:
        return sorted((path.name for path in self.albums_root.iterdir()
                       if path.is_dir() and not path.is_symlink()), key=str.casefold)

    def create_album(self, name: str) -> str:
        name = self._safe_album_name(name)
        path = self.albums_root / name
        if any(existing.casefold() == name.casefold() for existing in self.albums()):
            raise FileExistsError(f"Album already exists: {name}")
        path.mkdir()
        return name

    def rename_album(self, old_name: str, new_name: str) -> str:
        old_name = self._safe_album_name(old_name)
        new_name = self._safe_album_name(new_name)
        if old_name.casefold() == new_name.casefold():
            return old_name
        source = self.albums_root / old_name
        destination = self.albums_root / new_name
        if not source.is_dir():
            raise FileNotFoundError(f"Album does not exist: {old_name}")
        if any(existing.casefold() == new_name.casefold() for existing in self.albums()):
            raise FileExistsError(f"Album already exists: {new_name}")
        source.rename(destination)
        try:
            with self._db() as db:
                db.execute("UPDATE assets SET path = replace(path, ?, ?) WHERE path LIKE ?",
                           (f"library/Albums/{old_name}/", f"library/Albums/{new_name}/",
                            f"library/Albums/{old_name}/%"))
        except Exception:
            destination.rename(source)
            raise
        return new_name

    def register_image(self, path: Path, *, asset_id: str | None = None,
                       origin: str = "generated", run_id: str | None = None,
                       created_at: str | None = None, display_name: str | None = None,
                       prompt: str = "", model: str = "", tags=()) -> str:
        """Register an image already stored under Inbox or one of the albums."""
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError("A supported image file is required.")
        if not path.is_relative_to(self.library_root.resolve()):
            raise ValueError("Registered images must be inside the managed library.")
        relative = path.relative_to(self.output_root).as_posix()
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read image: {path.name}") from exc
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        asset_id = asset_id or uuid.uuid4().hex
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            db.execute("""INSERT INTO assets
                (asset_id, path, origin, run_id, created_at, display_name, prompt,
                 model, checksum, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (asset_id, relative, origin, run_id, created_at,
                 display_name or path.name, prompt, model, checksum, width, height))
            for tag in self._tag_list(tags):
                db.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", (tag,))
                db.execute("INSERT OR IGNORE INTO asset_tags(asset_id, tag) VALUES (?, ?)",
                           (asset_id, tag))
        return asset_id

    def import_image(self, source, *, tags=()) -> str:
        """Copy a path or uploaded image into Inbox and register the managed copy."""
        source_path = Path(source) if isinstance(source, (str, Path)) else None
        display_name = (source_path.name if source_path else
                        str(getattr(source, "name", "imported.png")))
        suffix = Path(display_name).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError("Only PNG, JPG, JPEG, and WebP images can be imported.")
        asset_id = uuid.uuid4().hex
        destination = self.inbox / f"{asset_id}{suffix}"
        if source_path is not None:
            if not source_path.is_file():
                raise ValueError("The image to import does not exist.")
            shutil.copy2(source_path, destination)
        else:
            payload = source.getvalue() if hasattr(source, "getvalue") else source.read()
            destination.write_bytes(payload)
        try:
            return self.register_image(destination, asset_id=asset_id, origin="imported",
                                       display_name=display_name, tags=tags)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def get_asset(self, asset_id: str) -> LibraryAsset:
        with self._db() as db:
            row = db.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown library image: {asset_id}")
            tags = [item[0] for item in db.execute(
                "SELECT tag FROM asset_tags WHERE asset_id = ? ORDER BY tag COLLATE NOCASE",
                (asset_id,))]
        return LibraryAsset(
            asset_id=row["asset_id"], path=self._asset_path(row["path"]),
            origin=row["origin"], run_id=row["run_id"], created_at=row["created_at"],
            display_name=row["display_name"], prompt=row["prompt"], model=row["model"],
            tags=tuple(tags), width=row["width"], height=row["height"],
        )

    def list_assets(self, *, query: str = "", album: str | None = None,
                    tags=(), offset: int = 0, limit: int = DEFAULT_PAGE_SIZE) -> list[LibraryAsset]:
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("Page offset must be nonnegative and page size must be 1–200.")
        filters = []
        values: list[object] = []
        if album:
            if album.casefold() == "inbox":
                prefix = "library/Inbox/"
            else:
                prefix = f"library/Albums/{self._safe_album_name(album)}/"
            filters.append("substr(a.path, 1, length(?)) = ?")
            values.extend([prefix, prefix])
        text = query.strip()
        if text:
            needle = f"%{text}%"
            filters.append("""(a.display_name LIKE ? OR a.prompt LIKE ? OR a.model LIKE ?
                            OR EXISTS (SELECT 1 FROM asset_tags at JOIN tags t ON t.tag = at.tag
                                       WHERE at.asset_id = a.asset_id AND t.tag LIKE ?))""")
            values.extend([needle, needle, needle, needle])
        requested_tags = self._tag_list(tags)
        for tag in requested_tags:
            filters.append("""EXISTS (SELECT 1 FROM asset_tags at JOIN tags t ON t.tag = at.tag
                                      WHERE at.asset_id = a.asset_id AND t.tag = ? COLLATE NOCASE)""")
            values.append(tag)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        sql = f"""SELECT a.* FROM assets a{where}
                  ORDER BY a.created_at DESC, a.asset_id DESC LIMIT ? OFFSET ?"""
        values.extend([limit, offset])
        with self._db() as db:
            rows = db.execute(sql, values).fetchall()
        return [self.get_asset(row["asset_id"]) for row in rows]

    def count_assets(self, *, query: str = "", album: str | None = None, tags=()) -> int:
        filters = []
        values: list[object] = []
        if album:
            if album.casefold() == "inbox":
                prefix = "library/Inbox/"
            else:
                prefix = f"library/Albums/{self._safe_album_name(album)}/"
            filters.append("substr(a.path, 1, length(?)) = ?")
            values.extend([prefix, prefix])
        text = query.strip()
        if text:
            needle = f"%{text}%"
            filters.append("""(a.display_name LIKE ? OR a.prompt LIKE ? OR a.model LIKE ?
                            OR EXISTS (SELECT 1 FROM asset_tags at JOIN tags t ON t.tag = at.tag
                                       WHERE at.asset_id = a.asset_id AND t.tag LIKE ?))""")
            values.extend([needle, needle, needle, needle])
        for tag in self._tag_list(tags):
            filters.append("""EXISTS (SELECT 1 FROM asset_tags at JOIN tags t ON t.tag = at.tag
                                      WHERE at.asset_id = a.asset_id AND t.tag = ? COLLATE NOCASE)""")
            values.append(tag)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._db() as db:
            return db.execute(f"SELECT COUNT(*) FROM assets a{where}", values).fetchone()[0]

    def all_tags(self) -> list[str]:
        with self._db() as db:
            return [row[0] for row in db.execute("SELECT tag FROM tags ORDER BY tag COLLATE NOCASE")]

    def set_tags(self, asset_ids, tags) -> None:
        values = self._tag_list(tags)
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return
        with self._db() as db:
            existing = {row[0] for row in db.execute(
                f"SELECT asset_id FROM assets WHERE asset_id IN ({','.join('?' for _ in ids)})", ids)}
            if existing != set(ids):
                raise KeyError("One or more selected library images no longer exist.")
            db.execute(f"DELETE FROM asset_tags WHERE asset_id IN ({','.join('?' for _ in ids)})", ids)
            for tag in values:
                db.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", (tag,))
                db.executemany("INSERT INTO asset_tags(asset_id, tag) VALUES (?, ?)",
                               [(asset_id, tag) for asset_id in ids])
            db.execute("DELETE FROM tags WHERE tag NOT IN (SELECT tag FROM asset_tags)")

    def move_assets(self, asset_ids, album: str | None = None) -> None:
        """Move selected images to Inbox (None) or a physical album."""
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return
        destination_root = self.inbox if album is None or album.casefold() == "inbox" else (
            self.albums_root / self._safe_album_name(album))
        destination_root.mkdir(parents=True, exist_ok=True)
        completed = []
        try:
            with self._db() as db:
                placeholders = ",".join("?" for _ in ids)
                rows = db.execute(f"SELECT * FROM assets WHERE asset_id IN ({placeholders})", ids).fetchall()
                if len(rows) != len(ids):
                    raise KeyError("One or more selected library images no longer exist.")
                moves = []
                for row in rows:
                    source = self._asset_path(row["path"])
                    if not source.is_file():
                        raise FileNotFoundError(f"Library image is missing: {source}")
                    destination = destination_root / source.name
                    if destination.exists() and destination.resolve() != source.resolve():
                        raise FileExistsError(f"Image already exists in destination: {destination.name}")
                    moves.append((row["asset_id"], source, destination))
                for asset_id, source, destination in moves:
                    if source != destination:
                        source.rename(destination)
                        completed.append((source, destination))
                    relative = destination.relative_to(self.output_root).as_posix()
                    db.execute("UPDATE assets SET path = ? WHERE asset_id = ?", (relative, asset_id))
        except Exception:
            for source, destination in reversed(completed):
                if destination.exists():
                    destination.rename(source)
            raise
        self._invalidate_thumbnails(ids)

    def add_tags(self, asset_ids, tags) -> None:
        ids = list(dict.fromkeys(asset_ids))
        new_tags = self._tag_list(tags)
        if not ids or not new_tags:
            return
        with self._db() as db:
            placeholders = ",".join("?" for _ in ids)
            existing = {row[0] for row in db.execute(
                f"SELECT asset_id FROM assets WHERE asset_id IN ({placeholders})", ids)}
            if existing != set(ids):
                raise KeyError("One or more selected library images no longer exist.")
            for tag in new_tags:
                db.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", (tag,))
                db.executemany("INSERT OR IGNORE INTO asset_tags(asset_id, tag) VALUES (?, ?)",
                               [(asset_id, tag) for asset_id in ids])

    def remove_assets(self, asset_ids) -> int:
        """Delete selected managed image files and catalog records."""
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return 0
        assets = [self.get_asset(asset_id) for asset_id in ids]
        staged = []
        try:
            for asset in assets:
                if asset.path.exists():
                    temporary = asset.path.with_name(asset.path.name + ".deleting")
                    asset.path.rename(temporary)
                    staged.append((asset.path, temporary))
            with self._db() as db:
                db.executemany("DELETE FROM assets WHERE asset_id = ?", [(item,) for item in ids])
        except Exception:
            for original, temporary in reversed(staged):
                if temporary.exists():
                    temporary.rename(original)
            raise
        for _, temporary in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # The committed deletion is complete; leave a recoverable hidden file
                # rather than restoring it without a catalog row.
                pass
        with self._db() as db:
            db.execute("DELETE FROM tags WHERE tag NOT IN (SELECT tag FROM asset_tags)")
        self._invalidate_thumbnails(ids)
        return len(ids)

    def thumbnail(self, asset_id: str, size: tuple[int, int] = (360, 360)) -> Path:
        asset = self.get_asset(asset_id)
        target = self.thumbnail_root / f"{asset_id}.jpg"
        if not target.exists() or target.stat().st_mtime < asset.path.stat().st_mtime:
            with Image.open(asset.path) as image:
                fitted = ImageOps.contain(image.convert("RGB"), size)
                canvas = Image.new("RGB", size, "#161616")
                canvas.paste(fitted, ((size[0] - fitted.width) // 2,
                                      (size[1] - fitted.height) // 2))
                canvas.save(target, format="JPEG", quality=85, optimize=True)
        return target

    def _invalidate_thumbnails(self, asset_ids) -> None:
        for asset_id in asset_ids:
            (self.thumbnail_root / f"{asset_id}.jpg").unlink(missing_ok=True)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        temporary.replace(path)

    def migrate_legacy_outputs(self, *, best_folder: str = "best") -> dict:
        """Move legacy run images and metadata into the library layout transactionally."""
        best_folder = self._safe_album_name(best_folder)
        run_root = self.output_root / "runs"

        source_manifests = []
        run_manifests: dict[str, dict] = {}
        image_plans = []
        seen_sources = set()
        skipped_metadata = 0
        for manifest_path in self.output_root.rglob("metadata.json"):
            parent = manifest_path.parent.resolve()
            if (parent.is_relative_to(run_root.resolve())
                    or parent.is_relative_to(self.library_root.resolve())):
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                skipped_metadata += 1
                continue
            if not isinstance(data, dict) or not isinstance(data.get("images"), list):
                skipped_metadata += 1
                continue
            images = []
            for name in data["images"]:
                if not isinstance(name, str) or Path(name).name != name:
                    continue
                source = (manifest_path.parent / name).resolve()
                if (source in seen_sources or not source.is_file()
                        or not source.is_relative_to(self.output_root)
                        or source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES):
                    continue
                seen_sources.add(source)
                images.append((name, source))
            if not images:
                continue

            best_of = data.get("best_of") if isinstance(data.get("best_of"), dict) else None
            if best_of and isinstance(best_of.get("run"), str) and best_of["run"].strip():
                run_id = self._safe_run_id(best_of["run"])
                album = best_folder
            else:
                run_id = self._safe_run_id(manifest_path.parent.name)
                album = None
            if run_id not in run_manifests:
                base = dict(data)
                base["images"] = []
                base["asset_ids"] = []
                base.update(schema_version=4, run_id=run_id)
                run_manifests[run_id] = base
            target_manifest = run_root / run_id / "metadata.json"
            source_manifests.append((manifest_path, data))
            for display_name, source in images:
                asset_id = uuid.uuid4().hex
                destination_root = self.inbox if album is None else self.albums_root / album
                destination = destination_root / f"{asset_id}{source.suffix.lower()}"
                image_plans.append({
                    "asset_id": asset_id,
                    "source": source,
                    "destination": destination,
                    "display_name": display_name,
                    "run_id": run_id,
                    "album": album,
                    "data": data,
                })
                run_manifests[run_id]["images"].append(display_name)
                run_manifests[run_id]["asset_ids"].append(asset_id)

        if not image_plans:
            return {"migrated_images": 0, "migrated_runs": 0,
                    "skipped_metadata": skipped_metadata}

        target_manifests = {run_id: run_root / run_id / "metadata.json"
                            for run_id in run_manifests}
        collisions = [str(path) for path in target_manifests.values() if path.exists()]
        collisions.extend(str(item["destination"]) for item in image_plans
                          if item["destination"].exists())
        if collisions:
            raise FileExistsError("Migration destination already exists: " + collisions[0])

        created_best_album = False
        if (any(item["album"] == best_folder for item in image_plans)
                and not any(item.casefold() == best_folder.casefold()
                            for item in self.albums())):
            self.create_album(best_folder)
            created_best_album = True

        journal_path = self.library_root / ".migration-journal.json"
        journal = {
            "images": [{"source": str(item["source"]),
                        "destination": str(item["destination"]),
                        "asset_id": item["asset_id"]} for item in image_plans],
            "manifests": [str(path) for path in target_manifests.values()],
            "source_manifests": [
                {"path": str(path), "data": data} for path, data in source_manifests
            ],
        }
        self._write_json(journal_path, journal)
        moved = []
        registered = []
        written_manifests = []
        created_run_dirs = []
        try:
            for item in image_plans:
                item["destination"].parent.mkdir(parents=True, exist_ok=True)
                expected = hashlib.sha256(item["source"].read_bytes()).hexdigest()
                item["source"].rename(item["destination"])
                moved.append((item["source"], item["destination"]))
                actual = hashlib.sha256(item["destination"].read_bytes()).hexdigest()
                if actual != expected:
                    raise OSError(f"Image verification failed: {item['display_name']}")
                self.register_image(
                    item["destination"], asset_id=item["asset_id"], origin="generated",
                    run_id=item["run_id"],
                    created_at=str(item["data"].get("created_at") or
                                   datetime.now(timezone.utc).isoformat()),
                    display_name=item["display_name"],
                    prompt=str(item["data"].get("prompt", "")),
                    model=str(item["data"].get("model", "")),
                )
                registered.append(item["asset_id"])

            for run_id, manifest in run_manifests.items():
                target = target_manifests[run_id]
                if not target.parent.exists():
                    target.parent.mkdir(parents=True)
                    created_run_dirs.append(target.parent)
                self._write_json(target, manifest)
                written_manifests.append(target)

            # Remove legacy manifests only after all new files and catalog rows verify.
            for old_manifest, _ in source_manifests:
                old_manifest.unlink()
        except Exception:
            for old_manifest, old_data in source_manifests:
                if not old_manifest.exists():
                    self._write_json(old_manifest, old_data)
            for path in reversed(written_manifests):
                path.unlink(missing_ok=True)
            if registered:
                with self._db() as db:
                    db.executemany("DELETE FROM assets WHERE asset_id = ?",
                                   [(asset_id,) for asset_id in registered])
            for source, destination in reversed(moved):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
            for directory in reversed(created_run_dirs):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            if created_best_album:
                try:
                    (self.albums_root / best_folder).rmdir()
                except OSError:
                    pass
            journal_path.unlink(missing_ok=True)
            raise
        else:
            journal_path.unlink(missing_ok=True)

        return {"migrated_images": len(image_plans),
                "migrated_runs": len(run_manifests),
                "skipped_metadata": skipped_metadata}

    def _recover_migration(self) -> None:
        """Roll back a migration left incomplete by a process or machine failure."""
        journal_path = self.library_root / ".migration-journal.json"
        if not journal_path.is_file():
            return
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            if not isinstance(journal, dict):
                raise ValueError("Migration journal is not an object.")
            image_moves = journal.get("images", [])
            run_manifests = journal.get("manifests", [])
            source_manifests = journal.get("source_manifests", [])
            if not all(isinstance(item, dict) for item in image_moves):
                raise ValueError("Migration journal has invalid image entries.")
            if not all(isinstance(item, str) for item in run_manifests):
                raise ValueError("Migration journal has invalid run manifest entries.")
            if not all(isinstance(item, dict) for item in source_manifests):
                raise ValueError("Migration journal has invalid source manifest entries.")

            root = self.output_root.resolve()
            library_root = self.library_root.resolve()
            resolved_moves = []
            asset_ids = []
            for item in image_moves:
                source = Path(item["source"]).resolve()
                destination = Path(item["destination"]).resolve()
                asset_id = str(item["asset_id"])
                if (not source.is_relative_to(root)
                        or not destination.is_relative_to(library_root)):
                    raise ValueError("Migration journal contains a path outside outputs.")
                resolved_moves.append((source, destination))
                asset_ids.append(asset_id)
            resolved_manifests = [Path(item).resolve() for item in run_manifests]
            runs_root = (root / "runs").resolve()
            if any(not path.is_relative_to(runs_root) for path in resolved_manifests):
                raise ValueError("Migration journal contains a run path outside outputs/runs.")
            resolved_sources = []
            for item in source_manifests:
                path = Path(item["path"]).resolve()
                if not path.is_relative_to(root) or path.is_relative_to(library_root):
                    raise ValueError("Migration journal contains an invalid legacy manifest path.")
                if not isinstance(item.get("data"), dict):
                    raise ValueError("Migration journal is missing a source manifest backup.")
                resolved_sources.append((path, item["data"]))

            for source, destination in reversed(resolved_moves):
                if destination.exists():
                    if source.exists():
                        raise FileExistsError(f"Both migration paths exist: {source}")
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
            if asset_ids:
                with self._db() as db:
                    db.executemany("DELETE FROM assets WHERE asset_id = ?",
                                   [(asset_id,) for asset_id in asset_ids])
                    db.execute("DELETE FROM tags WHERE tag NOT IN (SELECT tag FROM asset_tags)")
            for path in resolved_manifests:
                path.unlink(missing_ok=True)
            for path, data in resolved_sources:
                if not path.exists():
                    self._write_json(path, data)
            for path in resolved_manifests:
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
            journal_path.unlink()
        except Exception as exc:
            raise RuntimeError(
                f"Could not recover the interrupted image migration; journal retained at {journal_path}: {exc}"
            ) from exc

    @staticmethod
    def _safe_run_id(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:96]
        return slug or f"legacy_{uuid.uuid4().hex[:12]}"


def open_library(output_root: Path) -> Library:
    return Library(output_root)
