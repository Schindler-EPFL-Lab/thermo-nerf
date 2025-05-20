import json
from pathlib import Path
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Colormap
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline

from thermo_nerf.rendered_image_modalities import RenderedImageModality


class Renderer:
    """
    Renderer class aimed at being a cleaner replacement of the method
    `_render_trajectory_video` in the script `render.py` of nerfstudio
    """

    def __init__(self, model: Model) -> None:
        self._rendered_images: dict[RenderedImageModality, list[np.ndarray]] = {}
        self._model = model

    @property
    def model(self) -> Model:
        return self._model

    @staticmethod
    def _get_trainer(
        model_path: Path, eval_num_rays_per_chunk: Optional[int] = None
    ) -> TrainerConfig:
        """Get the trainer config associated to the model in `model_path`.

        `eval_num_rays_per_chunk` represents the number of rays to render per forward
        pass and a default value should exist in the loaded config file. Only change
        from `None` if the PC's memory can't handle rendering the default chunck / batch
        value per one forward pass.

        :raises RuntimeError: if more than one config can be found (recursively) in the
        path `model_path`
        :return: the trainer config that correspond to the model at `model_path`
        """
        render_config_paths = list(model_path.rglob("config.yml"))
        if len(render_config_paths) > 1:
            raise RuntimeError(
                "Try to load a model from a path where multiple models "
                "can be (recursively) found. Limit the path to a single "
                "model."
            )

        if len(render_config_paths) == 0:
            raise RuntimeError("No model found at path", model_path)

        with open(render_config_paths[0], "r") as stream:
            config = yaml.load(stream, Loader=yaml.Loader)
        assert isinstance(config, TrainerConfig)

        if eval_num_rays_per_chunk is not None:
            config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

        return config

    @staticmethod
    def extract_pipeline(
        model_path: Path,
        transforms_path: Path,
        eval_num_rays_per_chunk: int | None = None,
        test_mode: str = "inference",
        output_dir: Path | None = Path("."),
    ) -> tuple[Pipeline, TrainerConfig, int]:
        """
        Extract the pipeline from a model.

        Provide the `tranforms_path` and the `model_path` to load the pipeline.
        You can choose the `test_mode` in function of what you will do with the
        pipeline: if you need to run the model on data from the train dataset,
        use "test", otherwise "inference" is probably enough.
        `output_dir` is overwriting the output_dir of the configuration loaded to avoid
        inexistant paths. if set to None, the output_dir in the config file is not
        changed.

        :return: the pipeline and associated trainer config
        """
        config = Renderer._get_trainer(model_path, eval_num_rays_per_chunk)

        # Get model state
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = list(model_path.rglob("*.ckpt"))[-1]
        loaded_state = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Add an absolute data-asset path.
        config.pipeline.datamanager.data = transforms_path
        config.pipeline.datamanager.dataparser.data = transforms_path
        # Update output directory and make sure it exists
        if output_dir is not None:
            config.output_dir = output_dir
            config.get_base_dir().mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create pipeline from the config file content
        pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
        assert isinstance(pipeline, Pipeline)
        pipeline.eval()
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

        return pipeline, config, loaded_state["step"]

    @classmethod
    def from_pipeline_path(
        cls,
        model_path: Path,
        transforms_path: Path,
        eval_num_rays_per_chunk: Optional[int] = None,
    ) -> "Renderer":
        """Creates a renderer loading the model saved at `model_path`.

        `transforms_path` is a useless artifact from loading the pipeline instead of
        the model; it needs to point toward the transform of the generated data.
        `eval_num_rays_per_chunk` represents the number of rays to render per forward
        pass and a default value should exist in the loaded config file. Only change
        from `None` if the PC's memory can't handle rendering the default chunck / batch
        value per one forward pass.

        :returns: pipeline containg trained model with similar configurations to
        training.
        """
        pipeline, _, _ = Renderer.extract_pipeline(
            model_path=model_path,
            transforms_path=transforms_path,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        )
        return cls(pipeline.model)

    @staticmethod
    def load_cameras(
        load_camera_trajectory: Path,
        rendered_resolution_scaling_factor: float = 1.0,
    ) -> Cameras:
        """Load the path saved in the file `load_camera_trajectory` as Cameras instance
        The path will be scaled by `rendered_resolution_scaling_factor` which defaults
        to 1.0.

        :return: the path as Cameras instance.
        """
        with open(load_camera_trajectory, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        cameras = get_path_from_json(camera_path)
        cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
        return cameras

    def render(
        self,
        rendered_image_modalities: list[RenderedImageModality],
        cameras: Cameras,
        thermal_color_map: Colormap = plt.colormaps["magma"],
    ) -> None:
        """
        Helper function to create a video of trajectory of `cameras`.

        The video / frames of the trajectory generated through 'self.model', a trained
        checkpoint of the scene, and their modality is specified through
        `rendered_image_modalities`.

        the video is saved in `output_dir` with a resolution of
        `rendered_resolution_scaling_factor` * training image resolution.

        """
        cameras = cameras.to(self._model.device)

        self._rendered_images = {}
        for modality in rendered_image_modalities:
            self._rendered_images[modality] = []
            for camera_idx in range(cameras.size):
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
                with torch.no_grad():
                    outputs = self._model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle
                    )

                    if modality.value not in outputs:
                        raise Exception(f"{modality.value} modality does not exist")

                    output_image = outputs[modality.value].cpu().numpy()
                    if output_image.shape[-1] == 1:
                        output_image = np.concatenate((output_image,) * 3, axis=-1)
                    if modality == RenderedImageModality.THERMAL:
                        output_image = (
                            thermal_color_map(output_image[:, :, 0])[:, :, :3] * 255
                        ).astype(np.uint8)
                    else:
                        output_image = (output_image * 255).astype(np.uint8)
                    self._rendered_images[modality].append(output_image)

    def save_images(self, modalities: list[RenderedImageModality], output_dir: Path):
        """
        Saves the generated views of multiple images modalities specified through
        `modalities` as separate images in `output_dir`.
        """
        for modality in modalities:
            for idx, image in enumerate(self._rendered_images[modality]):
                imageio.imwrite(
                    (output_dir / f"{modality.value}_{idx:05d}.jpeg"),
                    image,
                )

    def save_gif(
        self, modalities: list[RenderedImageModality], seconds: float, output_dir: Path
    ):
        """
        Saves the generated views of multiple images modalities specified through
        `modalities` as a stitched video, of diration `seconds`, per modality in
        `output_dir`.
        """
        for modality in modalities:
            imageio.mimsave(
                output_dir / f"synthesized_video_{modality.value}.gif",
                self._rendered_images[modality],  # type: ignore
                duration=seconds,
            )
