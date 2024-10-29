import json
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from numpy.testing import assert_almost_equal

from thermo_scenes.flir_thermal_images.thermal_visualiser import ThermalVisualiser


class CustomThermalVisualiser(ThermalVisualiser):
    """
    This class inherits the ThermalVisualiser class but does not
    initialise the plotting metrics as they terminate the test prematurely.
    """

    def __init__(self, thermal_image, max_temperature, min_temperature) -> None:
        self.thermal_image = thermal_image
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature


class TestThermalConversion(unittest.TestCase):
    def test_temperature_conversion(self) -> None:
        gt_path = "tests/data/thermal/IMG_3561.csv"
        image_path = "tests/data/thermal/IMG_3561.PNG"
        bounds_path = "tests/data/thermal/temperature_bounds.json"

        with open(bounds_path, "r") as f:
            json_file = json.load(f)
            absolute_max_temperature = json_file["absolute_max_temperature"]
            absolute_min_temperature = json_file["absolute_min_temperature"]

        if absolute_max_temperature < absolute_min_temperature:
            absolute_max_temperature, absolute_min_temperature = (
                absolute_min_temperature,
                absolute_max_temperature,
            )

        gt_temperature = pd.read_csv(gt_path, usecols=["temp (c)"]).values
        gt_temperature = gt_temperature.reshape(640, 480)

        image = plt.imread(image_path)

        thermal_visualiser = CustomThermalVisualiser(
            image, absolute_max_temperature, absolute_min_temperature
        )

        for i in range(640):
            for j in range(480):
                pixel_value = image[i, j]
                temperature = thermal_visualiser.update_temperature(pixel_value)

                assert_almost_equal(temperature, gt_temperature[i, j], decimal=1)
