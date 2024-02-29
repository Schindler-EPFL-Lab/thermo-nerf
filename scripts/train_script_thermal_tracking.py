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

from nerfstudio.scripts.train import main  # noqa: E402

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
    output_dir: Path = "./outputs"
    """Where to save the model and outputs"""
    max_num_iterations: int = 30000
    data: Path = "./inputs"
    """Input data in azure format"""
    temperature_bounds: list = field(default_factory=lambda: [1.0, 0.0])
    """Temperature bounds for the dataset"""
    depth_scene_scale: float = 10.0
    """scaling factor to scale depth data to metric values"""
    use_uncertainty_loss: bool = True

    def __post_init__(self) -> None:
        mapping_name_to_config = {
            "thermal-nerf": ThermalNeRFTrackConfig,
        }
        self.model = mapping_name_to_config[self.model_type]


if __name__ == "__main__":
    parameters = tyro.cli(TrainingParameters)
    parameters.model.experiment_name = parameters.experiment_name
    parameters.model.output_dir = parameters.output_dir
    parameters.model.max_num_iterations = parameters.max_num_iterations
    parameters.model.data = parameters.data
    parameters.model.pipeline.model = ThermalNerfModelConfig(
        eval_num_rays_per_chunk=1 << 15,
        max_temperature=parameters.temperature_bounds[0],
        min_temperature=parameters.temperature_bounds[1],
        depth_scene_scale=parameters.depth_scene_scale,
        use_uncertainty_loss=parameters.use_uncertainty_loss,
    )
    parameters.model.viewer.quit_on_train_completion = True

    main(parameters.model)
