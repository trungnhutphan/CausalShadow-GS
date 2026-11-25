"""Acceptance checks for the NeRF authors' Lego example archive."""

import json
import math
from pathlib import Path
import unittest

import numpy as np
from PIL import Image


SCENE = Path(__file__).resolve().parents[1] / "data/raw/nerf_synthetic/lego"
FRAME_COUNTS = {"train": 100, "val": 100, "test": 200}


class LegoHandoffTest(unittest.TestCase):
    def test_camera_frames_and_color_images(self):
        for split, count in FRAME_COUNTS.items():
            metadata = json.loads((SCENE / f"transforms_{split}.json").read_text())
            self.assertTrue(0 < metadata["camera_angle_x"] < math.pi)
            self.assertEqual(len(metadata["frames"]), count)
            referenced = set()
            for frame in metadata["frames"]:
                with self.subTest(split=split, frame=frame["file_path"]):
                    path = Path(frame["file_path"] + ".png")
                    self.assertEqual(path.parent, Path(split))
                    self.assertRegex(path.name, r"^r_\d+\.png$")
                    self.assertNotIn(path, referenced)
                    referenced.add(path)
                    pose = np.asarray(frame["transform_matrix"], dtype=float)
                    self.assertEqual(pose.shape, (4, 4))
                    self.assertTrue(np.isfinite(pose).all())
                    np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                    rotation = pose[:3, :3]
                    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=5)
                    with Image.open(SCENE / path) as image:
                        image.load()
                        self.assertEqual((image.format, image.mode, image.size),
                                         ("PNG", "RGBA", (800, 800)))
            color_files = {p.relative_to(SCENE) for p in (SCENE / split).glob("*.png")
                           if p.stem.removeprefix("r_").isdigit()}
            self.assertEqual(referenced, color_files)

    def test_auxiliary_test_images(self):
        for kind in ("depth", "normal"):
            images = sorted((SCENE / "test").glob(f"r_*_{kind}_*.png"))
            self.assertEqual(len(images), FRAME_COUNTS["test"])
            for path in images:
                with self.subTest(path=path.name), Image.open(path) as image:
                    image.load()
                    self.assertEqual((image.format, image.size), ("PNG", (800, 800)))
