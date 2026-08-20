"""Fetch and prepare canonical tag36h11 PNGs for Isaac Sim textures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
from pupil_apriltags import Detector


SOURCE = (
    "https://raw.githubusercontent.com/AprilRobotics/"
    "apriltag-imgs/master/tag36h11/tag36_11_{tag_id:05d}.png"
)


def _detect(detector: Detector, image: Image.Image) -> list[int]:
    canvas = Image.new("L", (640, 640), 255)
    canvas.paste(image, ((640 - image.width) // 2, (640 - image.height) // 2))
    return [int(result.tag_id) for result in detector.detect(np.asarray(canvas))]


def main() -> None:
    output = Path("assets/tags")
    output.mkdir(parents=True, exist_ok=True)
    detector = Detector(families="tag36h11")
    for tag_id in range(30, 38):
        with urlopen(SOURCE.format(tag_id=tag_id), timeout=20) as response:
            source = Image.open(BytesIO(response.read())).convert("L")
        # Keep the official bit matrix intact and add the white quiet zone that
        # AprilTag detection requires. The whole texture maps to the 120 mm USD
        # panel, so this remains deterministic across local and remote runs.
        matrix = source.resize((400, 400), Image.Resampling.NEAREST)
        texture = Image.new("L", (512, 512), 255)
        texture.paste(matrix, (56, 56))
        detected = _detect(detector, texture)
        if detected != [tag_id]:
            raise RuntimeError(f"tag {tag_id} validation failed: detected {detected}")
        texture.save(output / f"tag36h11_{tag_id}.png")


if __name__ == "__main__":
    main()
