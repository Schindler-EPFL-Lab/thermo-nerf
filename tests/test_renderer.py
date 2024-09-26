import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from nerfstudio.models.vanilla_nerf import NeRFModel

from thermo_nerf.render.renderer import Renderer
from thermo_nerf.rendered_image_modalities import RenderedImageModality


class TestRenderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch(
            "nerfstudio.data.dataparsers.nerfstudio_dataparser"
            ".Nerfstudio._generate_dataparser_outputs"
        ):
            cls._renderer_vanilla = Renderer.from_pipeline_path(
                model_path=Path("tests/data/vanilla_nerf"),
                transforms_path=Path("tests/data/transforms.json"),
            )

        array = np.arange(0, 737280, 1, np.uint8)
        array = np.reshape(array, (1024, 720))

        cls._renderer_vanilla._rendered_images = {
            RenderedImageModality.RGB: [array, array]
        }

    def test_load(self) -> None:
        self.assertTrue(
            len(
                self._renderer_vanilla.model.state_dict()[
                    "field_coarse.mlp_base.layers.0.weight"
                ]
            ),
            256,
        )
        self.assertTrue(
            len(self._renderer_vanilla.model.state_dict()),
            71,
        )
        self.assertTrue(type(self._renderer_vanilla.model), NeRFModel)

    def test_image_export(self) -> None:
        output_dir = Path("tests/tmp_output")
        output_dir.mkdir(exist_ok=True)

        self._renderer_vanilla.save_images(
            [RenderedImageModality.RGB], output_dir=output_dir
        )
        files = []
        for file in output_dir.iterdir():
            files.append(file.name)
        self.assertTrue(len(files), 2)
        self.assertTrue("rgb_00000.jpeg" in files)
        self.assertTrue("rgb_00001.jpeg" in files)

    def test_gif_export(self) -> None:
        output_dir = Path("tests/tmp_output")
        output_dir.mkdir(exist_ok=True)
        self._renderer_vanilla.save_gif([RenderedImageModality.RGB], 1, output_dir)

    def test_load_cameras_positions(self) -> None:
        cameras = Renderer.load_cameras(
            Path("tests/data/trajectories/camera_path_facade_2.json")
        )
        self.assertTrue(cameras.camera_to_worlds.shape[0], 96)
