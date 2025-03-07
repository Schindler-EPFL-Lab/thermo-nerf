import json
import os
import shutil
import stat
from pathlib import Path


def thermal_as_rgb(thermal_dataset: Path, rgb_dataset: Path) -> None:
    """
    Copy the transforms.json in `rgb_dataset` to `thermal_dataset` and change
    the RGB path frame attribute by the thermal images path
    """
    shutil.copytree(src=rgb_dataset, dst=thermal_dataset)
    os.chmod(thermal_dataset / "transforms.json", stat.S_IRWXU)
    with open(thermal_dataset / "transforms.json", "r") as f:
        config = json.load(f)
    for frame in config["frames"]:
        frame["file_path"] = frame["thermal_file_path"]
    with open(thermal_dataset / "transforms.json", "w") as f:
        json.dump(config, f, indent=4)
