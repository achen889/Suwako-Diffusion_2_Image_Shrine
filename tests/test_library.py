import tempfile
import unittest
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from suwako.library import Library
from suwako.history import best_folder_name, rename_best_folder


class LibraryTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".test-tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temp.name)
        self.library = Library(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def make_image(self, name="source.png", color="green"):
        path = self.root / name
        Image.new("RGB", (40, 24), color).save(path)
        return path

    def test_import_copies_into_inbox_and_supports_tags_and_search(self):
        source = self.make_image()
        asset_id = self.library.import_image(source, tags=["pool", "Favorite"])

        asset = self.library.get_asset(asset_id)
        self.assertTrue(asset.path.is_relative_to(self.library.inbox))
        self.assertTrue(source.exists())
        self.assertEqual(asset.display_name, "source.png")
        self.assertEqual(asset.tags, ("Favorite", "pool"))
        self.assertEqual(self.library.count_assets(query="POOL"), 1)
        self.assertEqual(self.library.list_assets(album="Inbox")[0].asset_id, asset_id)

    def test_import_accepts_uploaded_file_and_preserves_display_name(self):
        payload = BytesIO()
        Image.new("RGB", (12, 9), "yellow").save(payload, format="PNG")

        class Upload:
            name = "uploaded.png"

            def getvalue(self):
                return payload.getvalue()

        asset_id = self.library.import_image(Upload())
        asset = self.library.get_asset(asset_id)
        self.assertEqual(asset.display_name, "uploaded.png")
        self.assertTrue(asset.path.is_file())
        self.assertEqual((asset.width, asset.height), (12, 9))

    def test_album_move_rename_and_tag_updates_keep_asset_id(self):
        asset_id = self.library.import_image(self.make_image())
        self.library.create_album("Favorites")
        self.library.move_assets([asset_id], "Favorites")
        asset = self.library.get_asset(asset_id)
        self.assertEqual(asset.path.parent.name, "Favorites")

        self.library.rename_album("Favorites", "Selected")
        self.assertEqual(self.library.get_asset(asset_id).path.parent.name, "Selected")
        self.library.add_tags([asset_id], ["portrait"])
        self.assertEqual(self.library.list_assets(tags=["portrait"])[0].asset_id, asset_id)
        self.library.move_assets([asset_id], "Inbox")
        self.assertEqual(self.library.get_asset(asset_id).path.parent.name, "Inbox")

    def test_renaming_best_album_updates_library_paths_and_setting(self):
        asset_id = self.library.import_image(self.make_image())
        self.library.create_album("best")
        self.library.move_assets([asset_id], "best")

        rename_best_folder("Favorites", self.root)

        self.assertEqual(best_folder_name(self.root), "Favorites")
        self.assertEqual(self.library.get_asset(asset_id).path.parent.name, "Favorites")

    def test_batch_move_rolls_back_files_when_a_move_fails(self):
        ids = [self.library.import_image(self.make_image(f"{n}.png", (n * 40, 0, 0)))
               for n in range(2)]
        sources = [self.library.get_asset(asset_id).path for asset_id in ids]
        original_rename = Path.rename

        def fail_on_second(path, target):
            if path == sources[1]:
                raise OSError("simulated move failure")
            return original_rename(path, target)

        self.library.create_album("Batch")
        with patch.object(Path, "rename", fail_on_second):
            with self.assertRaisesRegex(OSError, "simulated move failure"):
                self.library.move_assets(ids, "Batch")

        self.assertTrue(all(path.is_file() for path in sources))
        self.assertTrue(all(self.library.get_asset(asset_id).path.parent.name == "Inbox"
                            for asset_id in ids))

    def test_album_filter_treats_underscores_as_literal_characters(self):
        first = self.library.import_image(self.make_image("first.png"))
        second = self.library.import_image(self.make_image("second.png", "red"))
        self.library.create_album("Art_Box")
        self.library.create_album("ArtXBox")
        self.library.move_assets([first], "Art_Box")
        self.library.move_assets([second], "ArtXBox")

        self.assertEqual([asset.asset_id for asset in self.library.list_assets(album="Art_Box")],
                         [first])
        self.assertEqual(self.library.count_assets(album="Art_Box"), 1)

    def test_paging_and_thumbnail_cache(self):
        ids = [self.library.import_image(
            self.make_image(f"{n}.png", (n * 20, 10, 10))) for n in range(3)]
        self.assertEqual(self.library.count_assets(), 3)
        self.assertEqual(len(self.library.list_assets(offset=1, limit=1)), 1)
        thumb = self.library.thumbnail(ids[0])
        self.assertTrue(thumb.is_file())
        with Image.open(thumb) as image:
            self.assertEqual(image.size, (360, 360))

    def test_invalid_album_and_missing_asset_are_rejected(self):
        with self.assertRaises(ValueError):
            self.library.create_album("../outside")
        with self.assertRaises(KeyError):
            self.library.get_asset("missing")

    def test_legacy_migration_moves_images_and_keeps_run_provenance(self):
        run = self.root / "alexa" / "20261101T120000Z_pool_ab12cd34"
        run.mkdir(parents=True)
        image = run / "image_01.png"
        Image.new("RGB", (20, 10), "blue").save(image)
        (run / "metadata.json").write_text(json.dumps({
            "schema_version": 3, "created_at": "2026-11-01T12:00:00+00:00",
            "prompt": "indoor pool", "model": "animagine", "images": [image.name],
        }), encoding="utf-8")
        (self.root / "alexa" / "broken").mkdir()
        (self.root / "alexa" / "broken" / "metadata.json").write_text("not json", encoding="utf-8")

        result = self.library.migrate_legacy_outputs(best_folder="best")

        self.assertEqual(result["migrated_images"], 1)
        self.assertEqual(result["skipped_metadata"], 1)
        self.assertFalse(image.exists())
        run_data = json.loads((self.root / "runs" / run.name / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(len(run_data["asset_ids"]), 1)
        asset = self.library.get_asset(run_data["asset_ids"][0])
        self.assertEqual(asset.run_id, run.name)
        self.assertEqual(asset.prompt, "indoor pool")
        self.assertTrue(asset.path.is_relative_to(self.library.inbox))
        self.assertTrue((run / "metadata.json").is_file() is False)

    def test_legacy_migration_preflights_run_manifest_collision(self):
        run = self.root / "character" / "old-run"
        run.mkdir(parents=True)
        image = run / "image.png"
        Image.new("RGB", (10, 10), "red").save(image)
        (run / "metadata.json").write_text(json.dumps({"images": [image.name]}), encoding="utf-8")
        collision = self.root / "runs" / run.name
        collision.mkdir(parents=True)
        (collision / "metadata.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.library.migrate_legacy_outputs()

        self.assertTrue(image.is_file())
        self.assertEqual(list(self.library.inbox.iterdir()), [])
        self.assertEqual(self.library.albums(), [])

    def test_legacy_migration_rolls_back_file_moves_after_failure(self):
        run = self.root / "character" / "rollback-run"
        run.mkdir(parents=True)
        first = run / "image_01.png"
        second = run / "image_02.png"
        Image.new("RGB", (10, 10), "red").save(first)
        Image.new("RGB", (10, 10), "blue").save(second)
        manifest = run / "metadata.json"
        manifest.write_text(json.dumps({"images": [first.name, second.name]}), encoding="utf-8")
        original_rename = Path.rename

        def fail_on_second_image(path, target):
            if path == second:
                raise OSError("simulated disk error")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_on_second_image):
            with self.assertRaisesRegex(OSError, "simulated disk error"):
                self.library.migrate_legacy_outputs()

        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertTrue(manifest.is_file())
        self.assertEqual(self.library.count_assets(), 0)
        self.assertEqual(list(self.library.inbox.iterdir()), [])
        self.assertFalse((self.library.library_root / ".migration-journal.json").exists())

    def test_startup_recovers_an_interrupted_migration(self):
        run = self.root / "character" / "interrupted-run"
        run.mkdir(parents=True)
        source = run / "image.png"
        Image.new("RGB", (10, 10), "purple").save(source)
        source_data = {"prompt": "old prompt", "images": [source.name]}
        old_manifest = run / "metadata.json"
        old_manifest.write_text(json.dumps(source_data), encoding="utf-8")
        asset_id = "a" * 32
        destination = self.library.inbox / f"{asset_id}.png"
        source.rename(destination)
        self.library.register_image(destination, asset_id=asset_id, run_id=run.name)
        target_manifest = self.root / "runs" / run.name / "metadata.json"
        self.library._write_json(target_manifest, {"run_id": run.name, "asset_ids": [asset_id]})
        old_manifest.unlink()
        journal = {
            "images": [{"source": str(source), "destination": str(destination),
                        "asset_id": asset_id}],
            "manifests": [str(target_manifest)],
            "source_manifests": [{"path": str(old_manifest), "data": source_data}],
        }
        self.library._write_json(self.library.library_root / ".migration-journal.json", journal)

        recovered = Library(self.root)

        self.assertTrue(source.is_file())
        self.assertFalse(destination.exists())
        self.assertEqual(recovered.count_assets(), 0)
        self.assertEqual(json.loads(old_manifest.read_text(encoding="utf-8")), source_data)
        self.assertFalse(target_manifest.exists())
        self.assertFalse((self.library.library_root / ".migration-journal.json").exists())


if __name__ == "__main__":
    unittest.main()
