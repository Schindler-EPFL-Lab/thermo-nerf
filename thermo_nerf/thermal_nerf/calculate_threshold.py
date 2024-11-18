from pathlib import Path

import cv2
from nerfstudio.utils.io import load_from_json

from thermo_nerf.model_type import ModelType


def calculate_threshold(data: Path, model_type: ModelType) -> float:
    """
    Calculates the threshold that differentiates between
    the foreground and background of the thermal `data`.

    :return: the mean threshold value of the dataset.
    """

    if data.suffix == ".json":
        meta = load_from_json(data)
        data_dir = data.parent
    else:
        meta = load_from_json(data / "transforms.json")
        data_dir = data
    threshold_list = []

    path_key = "thermal_file_path"
    if model_type == ModelType.NERFACTO:
        path_key = "file_path"

    for frame in meta["frames"]:
        filepath = Path(frame[path_key])
        image = cv2.imread(str(data_dir / filepath), cv2.IMREAD_GRAYSCALE)
        threshold_temp, _ = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_list.append(threshold_temp)

    threshold = sum(threshold_list) / len(threshold_list)
    threshold = threshold / 255.0
    return threshold
