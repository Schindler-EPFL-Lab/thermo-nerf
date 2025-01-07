import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from nerfstudio.data.datamanagers.base_datamanager import \
    VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.scripts.train import main

from thermo_nerf.evaluator.evaluator import Evaluator
from thermo_nerf.model_type import ModelType
from thermo_nerf.nerfacto_config.config_nerfacto import thermalnerfacto_config
from thermo_nerf.render.renderer import Renderer
from thermo_nerf.rendered_image_modalities import RenderedImageModality
from thermo_nerf.rgb_concat.config_concat_nerfacto import concat_nerf_config
from thermo_nerf.thermal_nerf.calculate_threshold import calculate_threshold
from thermo_nerf.thermal_nerf.config_thermal_nerf import thermal_nerf_config
from thermo_nerf.thermal_nerf.thermal_nerf_model import ThermalNerfModelConfig


@dataclass
class TrainingParameters:
    model_type: ModelType = ModelType.THERMONERF
    """What NeRF model to train. Defaults to Nerfacto"""
    experiment_name: str = "nerfacto training"
    """Name of the model to train"""
    model_output_folder: Path = Path("./outputs")
    """Where to save the model and outputs"""
    max_num_iterations: int = 30000
    data: Path = Path("./inputs")
    """Input data in azure format"""

    metrics_output_folder: Path = Path("./outputs/")

    seed: int = 0
    """Seed for the random number generator"""
    temperature_bounds: list = field(default_factory=lambda: [1.0, 0.0])
    """Temperature bounds for the dataset"""

    cold: bool = False
    """Flag to use settings for cold temperatures"""
    camera_optimizer_mode: Literal["off", "SO3xR3", "SE3"] = "SO3xR3"
    """Pose optimization strategy to use. Recommended to be SO3xR3."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "filename"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """

    def threshold(self):
        return calculate_threshold(self.data, self.model_type)

    def __post_init__(self) -> None:
        if self.model_type == ModelType.THERMONERF:
            self.model = thermal_nerf_config
            self.modalities_to_save = [
                RenderedImageModality.RGB,
                RenderedImageModality.THERMAL,
                RenderedImageModality.THERMAL_COMBINED,
            ]

        if self.model_type == ModelType.NERFACTO:
            self.model = thermalnerfacto_config
            self.modalities_to_save = [
                RenderedImageModality.RGB,
            ]
        if self.model_type == ModelType.CONCATNERF:
            self.model = concat_nerf_config
            self.modalities_to_save = [
                RenderedImageModality.RGB,
            ]


if __name__ == "__main__":
    parameters = tyro.cli(TrainingParameters)

    if parameters.model_type == ModelType.NERFACTO:
        tmp_folder = Path("./data_folder/")
        shutil.copytree(src=parameters.data, dst=tmp_folder)
        os.chmod(tmp_folder / "transforms.json", stat.S_IRWXU)
        with open(tmp_folder / "transforms.json", "r") as f:
            config = json.load(f)
        for frame in config["frames"]:
            frame["file_path"] = frame["thermal_file_path"]
        with open(tmp_folder / "transforms.json", "w") as f:
            json.dump(config, f, indent=4)
        parameters.data = tmp_folder

    parameters.model.experiment_name = parameters.experiment_name
    parameters.model.output_dir = parameters.model_output_folder
    parameters.model.max_num_iterations = parameters.max_num_iterations
    parameters.model.data = parameters.data
    parameters.model.viewer.quit_on_train_completion = True
    assert isinstance(parameters.model.pipeline.model, ThermalNerfModelConfig)
    parameters.model.pipeline.model.max_temperature = parameters.temperature_bounds[0]
    parameters.model.pipeline.model.min_temperature = parameters.temperature_bounds[1]
    parameters.model.pipeline.model.cold = parameters.cold
    parameters.model.pipeline.model.camera_optimizer_mode = parameters.camera_optimizer_mode  # noqa: E501
    assert isinstance(parameters.model.pipeline.datamanager, VanillaDataManagerConfig)
    assert isinstance(parameters.model.pipeline.datamanager.dataparser, NerfstudioDataParserConfig)  # noqa: E501
    parameters.model.pipeline.datamanager.dataparser.eval_mode = parameters.eval_mode

    main(parameters.model)

    pipeline, config = Renderer.extract_pipeline(
        parameters.model.output_dir, parameters.data
    )
    evaluator = Evaluator(
        pipeline=pipeline,
        config=config,
        modalities_to_save=parameters.modalities_to_save,
        threshold=parameters.threshold(),
    )

    evaluator.save_metrics(output_folder=parameters.metrics_output_folder)
    evaluator.save_images(
        modalities=parameters.modalities_to_save,
        output_path=parameters.metrics_output_folder,
    )
