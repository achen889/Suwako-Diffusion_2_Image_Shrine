import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from suwako.models import ImageRequest
from suwako.prompts import compose
from suwako.workflow import run_generation


class FakeProvider:
    def generate(self, request):
        images = [Image.new("RGB", (64, 64), "green") for _ in range(request.count)]
        return images, {"seed": 123, "per_image_seeds": list(range(123, 123 + request.count)),
                        "device": "fake", "prompt_sent": request.prompt}


class SuwakoTests(unittest.TestCase):
    def test_animagine_enhancement(self):
        request = ImageRequest("green hair", negative_prompt="blurred")
        positive, negative = compose(request)
        self.assertIn("masterpiece", positive)
        self.assertIn("blurred", negative)

    def test_waifu_prefix(self):
        request = ImageRequest("forest", model="waifu", width=512, height=512)
        positive, _ = compose(request)
        self.assertTrue(positive.startswith("masterpiece, best quality, forest"))

    def test_openai_request_default_is_one_in_cli(self):
        # Avoid unexpectedly requesting four billable images from the cloud.
        from suwako.cli import main
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["suwako", "test", "--provider", "openai"]), \
             patch("suwako.cli.run_generation") as fn:
            main()
            self.assertEqual(fn.call_args.args[0].count, 1)

    def test_exact_prompt_unchanged(self):
        request = ImageRequest("green hair", mode="exact")
        positive, negative = compose(request)
        self.assertEqual(positive, "green hair")
        self.assertEqual(negative, "")

    def test_local_refusal_of_multi_references(self):
        request = ImageRequest("test", reference_images=(Path("missing.png"),))
        with self.assertRaises(ValueError):
            request.validate()

    def test_saved_image_count_and_metadata(self):
        request = ImageRequest("frog shrine", count=4)
        with tempfile.TemporaryDirectory() as tmp:
            directory = run_generation(request, Path(tmp), FakeProvider())
            files = sorted(directory.glob("image_*.png"))
            self.assertEqual(len(files), 4)
            data = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(data["count"], 4)
            self.assertEqual(data["per_image_seeds"], [123, 124, 125, 126])


if __name__ == "__main__":
    unittest.main()
