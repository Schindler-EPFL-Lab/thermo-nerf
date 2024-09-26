from dataclasses import dataclass
from pathlib import Path

import tyro
from nerfstudio.scripts.train import main

from thermo_nerf.evaluator.evaluator import Evaluator
from thermo_nerf.model_type import ModelType
from thermo_nerf.nerfacto_config.config_nerfacto import nerfacto_config
from thermo_nerf.render.renderer import Renderer
from thermo_nerf.rendered_image_modalities import RenderedImageModality
from thermo_nerf.rgb_concat.config_concat_nerfacto import concat_nerf_config
from thermo_nerf.thermal_nerf.config_thermal_nerf import thermal_nerftrack_config


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

    def __post_init__(self) -> None:
        if self.model_type == ModelType.THERMONERF:
            self.model = thermal_nerftrack_config
            self.modalities_to_save = [
                RenderedImageModality.RGB,
                RenderedImageModality.THERMAL,
                RenderedImageModality.THERMAL_COMBINED,
            ]

        if self.model_type == ModelType.NERFACTO:
            self.model = nerfacto_config
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

    parameters.model.experiment_name = parameters.experiment_name
    parameters.model.output_dir = parameters.model_output_folder
    parameters.model.max_num_iterations = parameters.max_num_iterations
    parameters.model.data = parameters.data
    parameters.model.viewer.quit_on_train_completion = True
    main(parameters.model)

    pipeline, config = Renderer.extract_pipeline(
        parameters.model.output_dir, parameters.data
    )
    evaluator = Evaluator(
        pipeline=pipeline,
        config=config,
        modalities_to_save=parameters.modalities_to_save,
    )

    evaluator.save_metrics(output_folder=parameters.metrics_output_folder)
    evaluator.save_images(
        modalities=parameters.modalities_to_save,
        output_path=parameters.metrics_output_folder,
    )
