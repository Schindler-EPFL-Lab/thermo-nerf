import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro


@dataclass(kw_only=True)
class EvaluatePrecisionScriptParameters:
    img_path: Path = Path("data/camera_precision_experiment.tar.xz")
    """
    Path to the folder containing the images.
    Either a folder of images or a tar.xz file (with suffix .txz) containing the images.
    """

    def __post_init__(self):
        if not self.img_path.exists():
            raise FileNotFoundError(f"Could not find file: {self.img_path}")

        if self.img_path.is_file() and self.img_path.suffix == ".txz":
            data_folder = self.img_path.parent / self.img_path.stem
            with tarfile.open(self.img_path, "r:xz") as tar_ref:
                tar_ref.extractall(self.img_path.parent)
            self.img_path = data_folder

        self.img_path = self.img_path / "thermal"
        json_file = self.img_path.parent / "temperature_bounds.json"
        with open(json_file, "r") as file:
            data = json.load(file)
        self.min_value = data["absolute_min_temperature"]
        self.max_value = data["absolute_max_temperature"]


def read_image(image_path: Path, min_value: float, max_value: float) -> np.ndarray:
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read")

    # Convert the image to float32 to avoid overflow during accumulation
    image = image.astype(np.float32)

    image = min_value + (image / 255.0) * (max_value - min_value)

    return image


def pixel_wise_mean(
    image_folder: Path, min_value: float, max_value: float
) -> np.ndarray:
    """
    For each pixel of all images in image_folder calculate the mean
    and standard deviation of the temperature.
    """
    total_sum = None
    image_count = 0
    for image_path in image_folder.iterdir():
        image = read_image(image_path, min_value, max_value)
        # Calculate the standard deviation
        # Accumulate the mean and standard deviation
        if total_sum is None:
            total_sum = image
        else:
            total_sum += image
        image_count += 1

    assert total_sum is not None
    # Calculate the pixel-wise mean
    pixel_wise_mean = total_sum / image_count

    return pixel_wise_mean


def calculate_pixel_wise_variance(
    image_path: Path, pixel_mean: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    # Read the image
    image = read_image(image_path, min_value, max_value)

    return (image - pixel_mean) ** 2


def pixel_wise_std(
    image_folder: Path, pixel_mean: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """
    For each pixel of all images in image_folder calculate the mean
    and standard deviation of the temperature.
    """
    total_squared_diff = None
    image_count = 0
    for image in image_folder.iterdir():
        std = calculate_pixel_wise_variance(image, pixel_mean, min_value, max_value)
        # Calculate the standard deviation
        # Accumulate the mean and standard deviation
        if total_squared_diff is None:
            total_squared_diff = std
        else:
            total_squared_diff += std
        image_count += 1
    assert total_squared_diff is not None
    return np.sqrt(total_squared_diff / image_count)


if __name__ == "__main__":
    """
    Calculate the mean and standard deviation of the temperature for each pixel of all
    images in img_path.

    `img_path` contains the thermal images extracted using the script
    thermoscenes_preprocess_thermal
    """
    parameters = tyro.cli(EvaluatePrecisionScriptParameters)

    mean = pixel_wise_mean(
        parameters.img_path, parameters.min_value, parameters.max_value
    )
    std = pixel_wise_std(
        parameters.img_path, mean, parameters.min_value, parameters.max_value
    )

    print(f"Mean of pixel-wise std: {np.mean(std)}")
