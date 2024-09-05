from dataclasses import dataclass, field
from pathlib import Path

import tyro
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.train import main
from thermo_nerf.evaluator.evaluator import Evaluator
from thermo_nerf.render.renderer import RenderedImageModality, Renderer
from thermo_nerf.rgb_concat.config_concat_nerfacto import ConcatNerfConfig
from thermo_nerf.thermal_nerf.config_thermal_nerf import ThermalNeRFTrackConfig


@dataclass
class TrainingParameters:
    model_type: str = "thermal-nerf"
    """What NeRF model to train. Defaults to Nerfacto"""
    experiment_name: str = "nerfacto training"
    """Name of the model to train"""
    model_output_folder: Path = Path("./outputs")
    """Where to save the model and outputs"""
    max_num_iterations: int = 30000
    data: Path = Path("./inputs")
    """Input data in azure format"""

    metrics_output_folder: Path = Path("./outputs/")

    modalities_to_save: list[RenderedImageModality] = field(
        default_factory=lambda: [
            RenderedImageModality.rgb,
        ]
    )
    """Name of the renderer outputs to use: rgb, depth, accumulation."""

    seed: int = 0
    """Seed for the random number generator"""

    def __post_init__(self) -> None:
        mapping_name_to_config = {
            "nerfacto": TrainerConfig,
            "thermal-nerf": ThermalNeRFTrackConfig,
            "concat-nerf": ConcatNerfConfig,
        }
        self.model = mapping_name_to_config[self.model_type]


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
    )

    evaluator.save_metrics(output_folder=parameters.metrics_output_folder)
    evaluator.save_images(
        modalities=parameters.modalities_to_save,
        output_path=parameters.metrics_output_folder,
    )
