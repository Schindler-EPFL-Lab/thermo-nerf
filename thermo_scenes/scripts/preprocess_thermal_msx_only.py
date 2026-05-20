"""
Preprocess RGB and thermal using MSX images only (it is not required to download RGB
images one by one from the FLIR website for this script).

IMPORTANT: The images cropped using this method still differ from those downloaded
individually from the FLIR website. There is a small rotation offset, which is
unfortunately unknown. Emperically found that the offset should be 0.165 or -0.165
degrees for horizontal images. The best way would be to process a few images and define
the sign.
"""

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
    angle: float = 0.165
    """
    Unknown angle to rotate the image before cropping in degrees. Empirically found that
    good values are either 0.165 or -0.165 for horizontal images.
    """


def main() -> None:
    paths = tyro.cli(Paths)

    _ = CustomFlir(
        path_to_msx_images=paths.msx_images,
        path_to_output_folder=paths.output_folder,
    )

    # from comments here
    # https://stackoverflow.com/questions/62095220/overlaping-raw-thermal-image-and-embedded-image-from-flir
    # but is seems that there is an (unknown) rotation between the downloaded
    # (one by one) and cropped frames :(
    cropped_rgb_folder = paths.output_folder / "cropped_rgb"
    cropped_rgb_folder.mkdir(exist_ok=True)
    for rgb_image in (paths.output_folder / "rgb").iterdir():
        offsets_json = subprocess.check_output(
            [
                "exiftool",
                "-OffsetX",
                "-OffsetY",
                "-Real2IR",
                "-EmbeddedImageWidth",
                "-EmbeddedImageHeight",
                "-j",
                paths.msx_images / (rgb_image.stem + ".JPG"),
            ]
        )
        offsets = json.loads(offsets_json.decode())[0]
        real_to_ir: float = offsets["Real2IR"]
        img_visual = Image.open(rgb_image)
        img_visual = img_visual.rotate(paths.angle)

        x_size = int(img_visual.size[0] * real_to_ir)
        y_size = int(img_visual.size[1] * real_to_ir)
        img_visual = img_visual.resize((x_size, y_size))
        crop_h: int = offsets["EmbeddedImageHeight"]
        crop_w: int = offsets["EmbeddedImageWidth"]

        start_y = (img_visual.size[1] - crop_h) // 2 + int(
            int(offsets["OffsetY"]) * real_to_ir
        )
        start_x = (img_visual.size[0] - crop_w) // 2 + int(
            int(offsets["OffsetX"]) * real_to_ir
        )

        img_visual = img_visual.crop(
            (
                start_x,
                start_y,
                start_x + crop_w,
                start_y + crop_h,
            )
        )
        img_visual.save(cropped_rgb_folder / rgb_image.name)


if __name__ == "__main__":
    main()
