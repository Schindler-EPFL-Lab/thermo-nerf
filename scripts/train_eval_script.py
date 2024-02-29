import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")

import mlflow
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.method_configs import method_configs  # noqa: E402
from nerfstudio.scripts.train import main  # noqa: E402
from rebel_nerf.evaluator.evaluator import Evaluator  # noqa: E402
from rebel_nerf.render.renderer import RenderedImageModality, Renderer  # noqa: E402
from rebel_nerf.rgb_concat.config_concat_nerfacto import ConcatNerfConfig
from rebel_nerf.semantic_sdf.base_models.config_nerfacto import (  # noqa: E402
    NeRFactoTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_neusfacto import (  # noqa: E402
    NeuSFactoTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_semantic_nerf import (  # noqa: E402
    SemanticNeRFTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_semantic_sdf import (  # noqa: E402
    SemanticSDFTrackConfig,
)
from rebel_nerf.thermal_nerf.config_thermal_nerf import (  # noqa: E402
    ThermalNeRFTrackConfig,
)
from rebel_nerf.uncertainty_nerf.config_uncertainty_nerfacto import (  # noqa: E402
    UncertaintyNerfConfig,
)
from rebel_nerf.uncertainty_nerf.create_noisy_dataset import transform_dataset
from rebel_nerf.uncertainty_nerf.uncertainty_nerf import UncertaintyNerfModelConfig


@dataclass
class TrainingParameters:
    data_asset_path: str
    """Path to save the dataset in a dataasset"""
    model_type: str = "nerfacto"
    """What NeRF model to train. Defaults to Nerfacto"""
    experiment_name: str = "nerfacto training"
    """Name of the model to train"""
    model_output_folder: Path = "./outputs"
    """Where to save the model and outputs"""
    max_num_iterations: int = 30000
    data: Path = "./inputs"
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
            "nerfacto": NeRFactoTrackConfig,
            "neus-facto": NeuSFactoTrackConfig,
            "semantic-nerf": SemanticNeRFTrackConfig,
            "semantic-sdf": SemanticSDFTrackConfig,
            "thermal-nerf": ThermalNeRFTrackConfig,
            "uncertainty-nerf": UncertaintyNerfConfig,
            "concat-nerf": ConcatNerfConfig,
        }
        self.model = mapping_name_to_config[self.model_type]


if __name__ == "__main__":
    parameters = tyro.cli(TrainingParameters)

    mlflow.log_param("model type", parameters.model_type)
    mlflow.log_param("max_num_iterations", parameters.max_num_iterations)

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
        job_param_identifier=parameters.job_param_identifier,
    )

    evaluator.save_metrics(output_folder=parameters.metrics_output_folder)
    evaluator.save_images(
        modalities=parameters.modalities_to_save,
        output_path=parameters.metrics_output_folder,
    )
