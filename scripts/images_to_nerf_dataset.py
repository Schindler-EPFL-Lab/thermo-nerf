"""
Processes an image sequence to a nerfstudio compatible dataset.
"""

import tyro
from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)


def main() -> None:
    """
    Processes images into a nerfstudio dataset.
    It calculates the camera poses for each image using `COLMAP`.
    """

    tyro.extras.set_accent_color("bright_yellow")
    process_imgs = tyro.cli(ImagesToNerfstudioDataset)
    process_imgs.main()


if __name__ == "__main__":
    main()
