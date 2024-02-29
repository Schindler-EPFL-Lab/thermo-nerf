"""
Train a radiance field with nerfstudio.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")

from nerfstudio.scripts.train import main  # noqa: E402

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


@dataclass
class TrainingParameters:
    model_type: str = "nerfacto"
    """What NeRF model to train. Defaults to Nerfacto"""
    experiment_name: str = "nerfacto training"
    """Name of the model to train"""
    output_dir: Path = "./outputs"
    """Where to save the model and outputs"""
    max_num_iterations: int = 30000
    data: Path = "./inputs"
    """Input data in azure format"""
    use_uncertainty_loss: bool = False

    def __post_init__(self) -> None:
        mapping_name_to_config = {
            "nerfacto": NeRFactoTrackConfig,
            "neus-facto": NeuSFactoTrackConfig,
            "semantic-nerf": SemanticNeRFTrackConfig,
            "semantic-sdf": SemanticSDFTrackConfig,
            "thermal-nerf": ThermalNeRFTrackConfig,
            "uncertainty-nerf": UncertaintyNerfConfig,
        }
        self.model = mapping_name_to_config[self.model_type]


if __name__ == "__main__":
    parameters = tyro.cli(TrainingParameters)

    parameters.model.experiment_name = parameters.experiment_name
    parameters.model.output_dir = parameters.output_dir
    parameters.model.max_num_iterations = parameters.max_num_iterations
    parameters.model.data = parameters.data
    parameters.model.viewer.quit_on_train_completion = True
    if parameters.model_type == "uncertainty-nerf":
        parameters.model.use_uncertainty_loss = parameters.use_uncertainty_loss  # type: ignore

    main(parameters.model)
