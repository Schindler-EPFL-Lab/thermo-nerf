"""
Processes an image sequence to a nerfstudio compatible dataset.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro
from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)

from thermo_scenes.update_colmap_json import update_colmap_json


@dataclass(kw_only=True)
class ImagesToNerfDatasetParams(ImagesToNerfstudioDataset):
    data: Path | None = None
    update_colmap_json: bool = True
    """ Whether to update json file with "thermal_file_path" arguments
    """
    thermo_scene_data: Path | None = None

    def __post_init__(self) -> None:

        if self.thermo_scene_data is not None:
            self.data = Path(self.thermo_scene_data, "rgb_train_processed")
            self.eval_data = Path(self.thermo_scene_data, "rgb_eval_processed")

        if self.data is None:
            raise RuntimeError(
                "self.data and self.thermo_scene_data cannot be both None"
            )

        return super().__post_init__()


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

    # Copy the thermal images
    if process_imgs.thermo_scene_data is None:
        return

    thermal_path = Path(process_imgs.output_dir, "thermal")
    thermal_path_train = Path(process_imgs.thermo_scene_data, "thermal_train_processed")
    thermal_path_eval = Path(process_imgs.thermo_scene_data, "thermal_eval_processed")
    thermal_path.mkdir(exist_ok=True, parents=True)

    for thermal_image in thermal_path_train.iterdir():
        shutil.copy(thermal_image, thermal_path)
    for thermal_image in thermal_path_eval.iterdir():
        shutil.copy(thermal_image, thermal_path)


if __name__ == "__main__":
    main()
