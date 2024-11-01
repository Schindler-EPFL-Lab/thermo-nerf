from dataclasses import dataclass
from pathlib import Path

import tyro

from thermo_scenes.flir_thermal_images.custom_flir import CustomFlir


@dataclass
class Paths:
    path_to_msx_images: Path
    """Path to the thermal data extracted from Flir One App."""
    path_to_output_folder: Path
    """Path to the output folder"""


def main() -> None:
    paths = tyro.cli(Paths)

    CustomFlir(
        path_to_msx_images=paths.path_to_msx_images,
        path_to_output_folder=paths.path_to_output_folder,
    )


if __name__ == "__main__":
    main()
