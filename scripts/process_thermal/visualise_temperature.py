import json
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed

sys.path.append(".")
sys.path.append("./nerfstudio")


from rebel_nerf.flir_themral_images.thermal_visualiser import (  # noqa: E402
    ThermalVisualiser,
)


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
