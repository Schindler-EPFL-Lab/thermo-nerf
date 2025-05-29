import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro
from PIL import Image

from thermo_scenes.flir_thermal_images.custom_flir import CustomFlir


@dataclass
class Paths:
    msx_images: Path = Path("data/datatest/")
    """Path to the thermal data extracted from Flir One App."""
    output_folder: Path = Path("data/output_datatest")
    """Path to the output folder"""


def main() -> None:
    paths = tyro.cli(Paths)

    _ = CustomFlir(
        path_to_msx_images=paths.msx_images,
        path_to_output_folder=paths.output_folder,
    )

    for rgb_image in (paths.output_folder / "rgb").iterdir():
        offsets_json = subprocess.check_output(
            [
                "exiftool",
                "-OffsetX",
                "-OffsetY",
                "-Real2IR",
                "-Megapixels",
                "-j",
                paths.msx_images / (rgb_image.stem + ".JPG"),
            ]
        )
        offsets = json.loads(offsets_json.decode())[0]
        real_to_ir: float = offsets["Real2IR"]
        scale: float = offsets["Megapixels"]
        img_visual = Image.open(rgb_image)
        x_size = int(img_visual.size[0] * scale * real_to_ir)
        y_size = int(img_visual.size[1] * scale * real_to_ir)
        img_visual = img_visual.resize((x_size, y_size))
        img_visual = img_visual.crop(
            (
                int(offsets["OffsetY"]),
                int(offsets["OffsetX"]),
                480 + int(offsets["OffsetY"]),
                640 + int(offsets["OffsetX"]),
            )
        )
        img_visual.save(rgb_image.parent / (rgb_image.stem + "cropped.png"))


if __name__ == "__main__":
    main()
