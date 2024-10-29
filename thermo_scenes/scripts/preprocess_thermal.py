from dataclasses import dataclass
from pathlib import Path

import tyro

from thermo_scenes.flir_thermal_images.custom_flir import CustomFlir


@dataclass
class Paths:
    path_to_thermal_images: str
    """Path to the thermal data extracted from Flir One App."""
    path_to_thermal_images_curated: str
    """Path to the output thermal images created using raw thermal data."""
    path_to_rgb: str
    """Path to the output rgb images"""
    path_to_csv_files: str
    """Path to the output csv files of temperature values"""
    path_to_txt: str = "temperature_bounds.json"
    """Path to the txt file with temperature bounds for the dataset"""


def main() -> None:
    paths = tyro.cli(Paths)
    list_thermal = []
    list_filenames = []
    flir = CustomFlir()
    for img_name in Path(paths.path_to_thermal_images).iterdir():
        flir.process_image(Path(paths.path_to_thermal_images, img_name.name))
        flir.export_thermal_to_csv(
            Path(paths.path_to_csv_files, str(img_name.name).split(".")[0] + ".csv")
        )
        flir.save_rgb_images(Path(paths.path_to_rgb))
        list_thermal.append(flir.get_thermal_np())
        list_filenames.append(str(img_name.name))

    flir.save_normalised_thermal_images(
        paths.path_to_thermal_images_curated, paths.path_to_csv_files
    )
    flir.save_temperature_bounds(paths.path_to_txt)


if __name__ == "__main__":
    main()
