import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from FlirImageExtractor.flir_image_extractor import FlirImageExtractor
from matplotlib import cm
from PIL import Image


def get_min_max_temperatures(path_to_csv_files: str) -> tuple[float, float]:
    """
    Extracts the minimum and maximum temperature value across all .csv files, found in
    'path_to_csv_files' directory, whereby each .csv file contains the thermal
    informaion of each pixel in the corresponding thermal image.

    :returns: minimum and maximum temperatures.
    """
    temperatures = []
    for path in Path(path_to_csv_files).iterdir():
        temperatures.extend(pd.read_csv(path, usecols=["temp (c)"]).dropna().values)

    absolute_min_temperature = min(temperatures).item()
    absolute_max_temperature = max(temperatures).item()
    return absolute_min_temperature, absolute_max_temperature


def normalise(
    input_temperatures: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    """
    Normalises the values of an `input` array corresponding to `minimum` and
    `maximum` respectively.

    :returns: normalised input between 0 and 1.
    """
    return (input_temperatures - minimum) / (maximum - minimum)


class CustomFlir(FlirImageExtractor):
    """
    Customised FlirImageExtractor class to extract thermal images from Flir One App.

    This class modofies the original FlirImageExtractor class mainly to save images
    independantly and to save all the thermal images in a dataset with a standardised
    colour map as they are normalised to the absolute maximum and minimum temperatures
    in the dataset.
    """

    def __init__(self, path_to_msx_images: Path, path_to_output_folder: Path) -> None:
        super().__init__()

        path_to_output_folder.mkdir(exist_ok=True, parents=True)
        if path_to_msx_images.parent != path_to_output_folder:
            shutil.copytree(
                path_to_msx_images,
                Path(path_to_output_folder, "msx"),
                dirs_exist_ok=True,
            )

        output_json_path = Path(path_to_output_folder, "temperature_bounds.json")

        output_rgb_folder = Path(path_to_output_folder, "rgb")
        output_rgb_folder.mkdir(parents=True, exist_ok=True)

        output_thermal_folder = Path(path_to_output_folder, "thermal")
        output_thermal_folder.mkdir(parents=True, exist_ok=True)

        output_csv_folder = Path(path_to_output_folder, "csv")
        output_csv_folder.mkdir(parents=True, exist_ok=True)

        self.absolute_max_temperature = None
        self.absolute_min_temperature = None

        for img_path in path_to_msx_images.iterdir():
            self.process_image(Path(path_to_msx_images, img_path.name))
            self.export_thermal_to_csv(Path(output_csv_folder, img_path.stem + ".csv"))
            self.save_rgb_images(output_rgb_folder)

        self.save_normalised_thermal_images(
            str(output_thermal_folder), str(output_csv_folder)
        )
        self.save_temperature_bounds(str(output_json_path))

    def save_rgb_images(self, path_to_rgb: Path) -> None:
        """
        Save the extracted rgb images only in the `path_to_rgb` directory.
        """
        rgb_np = self.get_rgb_np()

        img_visual = Image.fromarray(rgb_np)

        image_filename = Path(path_to_rgb, Path(self.flir_img_filename).stem + ".png")

        img_visual.save(image_filename)

    def save_normalised_thermal_images(
        self, path_to_thermal_images_curated: str, path_to_csv: str
    ) -> None:
        """
        Save the extracted thermal temperatures from all files in `path_to_csv` as
        normalised greyscale images.
        """
        (
            self.absolute_min_temperature,
            self.absolute_max_temperature,
        ) = get_min_max_temperatures(path_to_csv)

        for path in Path(path_to_csv).iterdir():
            temperature = pd.read_csv(path, usecols=["temp (c)"]).values
            temperature = 255 * normalise(
                temperature,
                self.absolute_min_temperature,
                self.absolute_max_temperature,
            )

            gray_scale_image = temperature.reshape(640, 480).astype("uint8")

            Image.fromarray(gray_scale_image, mode="L").save(
                Path(path_to_thermal_images_curated, Path(path).stem + ".png")
            )

    def save_normalised_coloured_thermal_images(
        self,
        path_to_thermal_images_curated: str,
        list_of_thermal_arrays: list[np.ndarray],
        list_filenames: list[str],
    ) -> None:
        """
        Save a coloured version of the thermal images that is based on a standard
        colour map by normalising the images to the absolute maximum and minimum
        temperatures across all images.

        The thermal images are encoded as numpy arrays, `list_of_thermal_arrays`,
        and saved in `path_to_thermal_images_curated`
        using the filenames in `list_filenames`.

        """
        self.absolute_max_temperature = max(
            [
                np.amax(thermal_array)
                for thermal_array in list_of_thermal_arrays
                if thermal_array is not None
            ]
        )
        self.absolute_min_temperature = min(
            [
                np.amin(thermal_array)
                for thermal_array in list_of_thermal_arrays
                if thermal_array is not None
            ]
        )

        for thermal_array, filename in zip(list_of_thermal_arrays, list_filenames):
            thermal_normalized = (thermal_array - self.absolute_min_temperature) / (
                self.absolute_max_temperature - self.absolute_min_temperature
            )
            img_thermal = Image.fromarray(
                np.uint8(cm.inferno(thermal_normalized) * 255)
            )

            thermal_filename = Path(
                path_to_thermal_images_curated, Path(filename).stem + ".png"
            )

            img_thermal.save(str(thermal_filename))

    def save_temperature_bounds(self, path_to_json: str) -> None:
        """
        Save the absolute maximum and minimum temperatures in a
        json file to be saved with the dataset.
        """

        with open(path_to_json, "w") as f:
            json.dump(
                {
                    "absolute_max_temperature": self.absolute_max_temperature,
                    "absolute_min_temperature": self.absolute_min_temperature,
                },
                f,
            )
