from pathlib import Path

import tyro

from thermo_scenes.update_colmap_json import update_colmap_json


def main(input_folder: Path) -> None:
    """
    Update the 'transforms.json' with new arguments 'thermal_file_path
    Set 'thermal_file_path' as "thermal/name", with the name in 'file_path''
    """

    update_colmap_json(input_folder=input_folder)


if __name__ == "__main__":
    tyro.cli(main)
