from dataclasses import dataclass
from pathlib import Path

import tyro

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


if __name__ == "__main__":
    main()
