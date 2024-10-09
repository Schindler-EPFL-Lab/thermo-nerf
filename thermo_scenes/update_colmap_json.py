import json
from pathlib import Path


def update_colmap_json(input_folder: Path) -> None:
    """
    Adds the attribute "thermal_file_path" to "transforms.json" in `input_folder`.

    The "thermal_file_path" is the same as "file_path" but in a folder named "thermal"
    instead of "images".
    """
    input_path = Path(input_folder, "transforms.json")

    with open(input_path, "r") as file:
        data = json.load(file)

    for frame in data["frames"]:
        file_path = Path(frame["file_path"])
        thermal_file_path = Path("thermal", file_path.name)
        frame["thermal_file_path"] = str(thermal_file_path)

    output_path = Path(input_folder, "transforms.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
