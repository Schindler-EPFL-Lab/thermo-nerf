import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig  # noqa: E402
from nerfstudio.scripts.train import main  # noqa: E402
from rebel_nerf.evaluator.evaluator import Evaluator  # noqa: E402
from rebel_nerf.render.renderer import RenderedImageModality, Renderer  # noqa: E402
from rebel_nerf.thermal_nerf.calculate_threshold import (  # noqa: E402
    calculate_threshold,
)
from rebel_nerf.thermal_nerf.config_thermal_nerf import (  # noqa: E402
    ThermalNeRFTrackConfig,
)
from rebel_nerf.thermal_nerf.thermal_nerf_model import (  # noqa: E402
    ThermalNerfModelConfig,
)


@dataclass
class TrainingParameters:
    model_type: str = "thermal-nerf"
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
            RenderedImageModality.thermal,
        ]
    )
    """Name of the renderer outputs to use: rgb, depth, accumulation."""

    temperature_bounds: list = field(default_factory=lambda: [1.0, 0.0])
    """Temperature bounds for the dataset"""

    def threshold(self):
        return calculate_threshold(self.data)

    def __post_init__(self) -> None:
        self.model = ThermalNeRFTrackConfig
        self.model.experiment_name = self.experiment_name
        self.model.output_dir = self.model_output_folder
        self.model.max_num_iterations = self.max_num_iterations
        self.model.data = self.data
        self.model.viewer.quit_on_train_completion = True
        self.model.pipeline.model = ThermalNerfModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            use_uncertainty_loss=self.use_uncertainty_loss,
            max_temperature=self.temperature_bounds[0],
            min_temperature=self.temperature_bounds[1],
            threshold=self.threshold(),
        )


if __name__ == "__main__":
    parameters = tyro.cli(TrainingParameters)

    mlflow.log_params(asdict(parameters))

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
