"""
Processes an image sequence to a nerfstudio compatible dataset.
"""

from dataclasses import dataclass

import tyro
from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)

from thermo_scenes.update_colmap_json import update_colmap_json


@dataclass
class ImagesToNerfDatasetParams(ImagesToNerfstudioDataset):
    update_colmap_json: bool = True
    """ Whether to update json file with "thermal_file_path" arguments
    """


def main() -> None:
    """
    Processes images into a nerfstudio dataset.
    It calculates the camera poses for each image using `COLMAP`.
    """

    tyro.extras.set_accent_color("bright_yellow")
    process_imgs = tyro.cli(ImagesToNerfDatasetParams)
    process_imgs.main()

    if process_imgs.update_colmap_json:
        update_colmap_json(input_folder=process_imgs.output_dir)


if __name__ == "__main__":
    main()
