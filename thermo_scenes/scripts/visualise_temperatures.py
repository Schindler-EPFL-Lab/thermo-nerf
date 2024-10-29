import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tyro

from thermo_scenes.flir_thermal_images.thermal_visualiser import ThermalVisualiser


@dataclass
class Paths:
    path_to_thermal_image: str
    """Path to one thermal image to visualise."""
    path_to_json_file: str
    """Path to the txt file with temperature bounds for the
    dataset of the image"""


def main() -> None:
    paths = tyro.cli(Paths)
    thermal_image = plt.imread(paths.path_to_thermal_image)
    print(thermal_image)

    with open(paths.path_to_json_file, "r") as f:
        json_file = json.load(f)
        absolute_max_temperature = json_file["absolute_max_temperature"]
        absolute_min_temperature = json_file["absolute_min_temperature"]

    if absolute_max_temperature < absolute_min_temperature:
        absolute_max_temperature, absolute_min_temperature = (
            absolute_min_temperature,
            absolute_max_temperature,
        )

    thermal_visualiser = ThermalVisualiser(
        thermal_image, absolute_max_temperature, absolute_min_temperature
    )
    thermal_visualiser.fig.canvas.mpl_connect(
        "motion_notify_event", thermal_visualiser.hover
    )
    plt.show()


if __name__ == "__main__":
    main()
