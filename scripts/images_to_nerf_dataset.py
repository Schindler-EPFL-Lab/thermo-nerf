"""
Processes an image sequence to a nerfstudio compatible dataset.
"""
import sys

import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path
sys.path.append(".")
sys.path.append("./nerfstudio")

from nerfstudio.process_data.images_to_nerfstudio_dataset import (  # noqa: E402
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
