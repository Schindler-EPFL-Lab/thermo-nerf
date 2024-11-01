from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image as PILImage
from PIL.Image import Image


@dataclass
class RescaleParams:
    input_folder: Path
    """Folder holding the thermal data"""
    output_folder: Path
    """Folder where the converted images in `self.input_path` will be saved"""
    t_min: float
    """Test image lower temperature bound"""
    t_max: float
    """Test image upper temperature bound"""
    t_min_new: float
    """ Train image lower temperature bound"""
    t_max_new: float
    """ Train image upper temperature bound"""


def scale(
    img_array: np.ndarray,
    min_origin: float,
    max_origin: float,
    min_target: float,
    max_target: float,
) -> np.ndarray:
    """
    Scales the values of `img_array` from `min_origin` and `max_origin` to
    `min_target` and `max_target`

    :returns: a scaled image in the range [`min_target`, `max_target`]
    """
    return (img_array - min_origin) / (max_origin - min_origin) * (
        max_target - min_target
    ) + min_target


def scale_image(img_array: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """
    Scale the `img_array` from [0, 1] to [`t_min`, `t_max`].

    :returns: a scaled image in the range [`t_min`, `t_max`]
    """

    return scale(
        img_array=img_array,
        min_origin=0,
        max_origin=1,
        min_target=t_min,
        max_target=t_max,
    )


def unscale_image(img_array: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """
    Scale the `img_array` array from [`t_min`, `t_max`] to [0, 1].

    :returns: an unscaled image in the range [0, 1]
    """

    return scale(
        img_array=img_array,
        min_origin=t_min,
        max_origin=t_max,
        min_target=0,
        max_target=1,
    )


def scale_test_to_train(
    img: Image,
    t_min: float,
    t_max: float,
    t_min_new: float,
    t_max_new: float,
) -> Image:
    """
    Make sure test images to be in the same temperature bounds as train images
    from `t_min` and `t_max` to `t_min_new`and `t_max_new`

    Original test image temperature range from `t_min` to `t_max`
    Then scaled to target train image temperature range from `t_min_new` to `t_max_new`

    :returns: a test Image scaled to the train image temparature range
    """
    # Convert to [0, 1]
    img_array = np.array(img) / 255.0

    img_array_scaled = scale_image(img_array, t_min, t_max)

    img_array_unscaled = unscale_image(img_array_scaled, t_min_new, t_max_new)

    # Ensure values are in [0, 1]
    img_array_unscaled = np.clip(img_array_unscaled, 0, 1)

    return PILImage.fromarray((img_array_unscaled * 255).astype(np.uint8))


def process_images(
    input_folder: Path,
    output_folder: Path,
    t_min: float,
    t_max: float,
    t_min_new: float,
    t_max_new: float,
) -> None:
    """
    Scales all images in the `input_folder` and saves them in the `output folder`

    Access all the test images from the path `input_folder`.
    Scales them from `t_min` and `t_max` to `t_min_new`and `t_max_new`
    Saves the scaled images unded the path `output_folder`
    """
    if not output_folder.exists():
        output_folder.mkdir()

    for filename in input_folder.iterdir():
        if filename.suffix not in [".PNG", ".JPG", ".jpeg", ".png", ".jpg"]:
            continue

        img_path = Path(input_folder, filename)
        img = PILImage.open(img_path).convert("RGB")

        img_final = scale_test_to_train(
            img=img, t_min=t_min, t_max=t_max, t_min_new=t_min_new, t_max_new=t_max_new
        )

        img_final.save(Path(output_folder, filename.name))


def main() -> None:
    params = tyro.cli(RescaleParams)

    process_images(
        params.input_folder,
        params.output_folder,
        params.t_min,
        params.t_max,
        params.t_min_new,
        params.t_max_new,
    )


if __name__ == "__main__":
    main()
