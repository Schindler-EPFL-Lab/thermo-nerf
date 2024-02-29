from pathlib import Path

import cv2
from nerfstudio.utils.io import load_from_json


def calculate_threshold(data: Path):
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
    for frame in meta["frames"]:
        filepath = Path(frame["thermal_file_path"])
        image = cv2.imread(str(data_dir / filepath), cv2.IMREAD_GRAYSCALE)
        threshold_temp, _ = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_list.append(threshold_temp)

    threshold = sum(threshold_list) / len(threshold_list)
    threshold = threshold / 255.0
    return threshold
