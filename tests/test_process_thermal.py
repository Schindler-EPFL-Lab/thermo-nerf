import shutil
import unittest
from pathlib import Path

from thermo_scenes.flir_thermal_images.custom_flir import CustomFlir


class TestThermalConversion(unittest.TestCase):
    def test_process_thermal(self) -> None:
        """
        Test of extracting RGB, thermal and csv files from the given msx images.

        2 msx images will generate 2 RGB PNG files, 2 thermal PNG files, 2 csv files
        and 1 temperature_bound.json file.
        """
        msx_path = Path("tests/data/process_thermal/msx")
        output_path = Path("tests/data/process_thermal/output")

        output_path.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        CustomFlir(path_to_msx_images=msx_path, path_to_output_folder=output_path)

        json_count = 0
        PNG_count = 0
        csv_count = 0

        for folder in output_path.iterdir():
            if folder.suffix == ".json":
                json_count += 1
                continue
            for files in folder.iterdir():
                if files.suffix in [".png", ".PNG", ".jpg", ".jpeg"]:
                    PNG_count += 1
                elif files.suffix == ".csv":
                    csv_count += 1

        self.assertEqual(json_count, 1, "The number of the JSON file is not correct!")
        self.assertEqual(PNG_count, 4, "The number of the PNG files is not correct!")
        self.assertEqual(csv_count, 2, "The number of the CSV files is not correct!")

    def test_name_consistency(self) -> None:
        """
        Test the name consistency of extracting RGB, thermal and csv files from the
        given msx images.

        RGB image, thermal image and csv file should have the same stem as the extracted
        msx image.
        """
        msx_path = Path("tests/data/process_thermal/msx")
        output_path = Path("tests/data/process_thermal/output")

        output_path.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        CustomFlir(path_to_msx_images=msx_path, path_to_output_folder=output_path)

        for folder in output_path.iterdir():
            if folder.suffix == ".json":
                continue
            namelist = [file.stem for file in msx_path.iterdir()]
            for files in folder.iterdir():
                if files.stem in namelist:
                    namelist.remove(files.stem)
            self.assertEqual(
                len(namelist),
                0,
                "Names of the extracted files are not consistent with the "
                + "given msx images!",
            )
