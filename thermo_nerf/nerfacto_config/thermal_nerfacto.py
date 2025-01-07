from dataclasses import dataclass, field
from typing import Literal, Type

import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from torch import Tensor

from thermo_nerf.thermal_nerf.thermal_metrics import mae_thermal


@dataclass
class ThermalNerfactoModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ThermalNerfactoModel)
    max_temperature: float = 1.0
    """Maximum temperature in the dataset."""
    min_temperature: float = 0.0
    """Minimum temperature in the dataset."""
    cold: bool = False
    """Flag to indicate if the dataset includes cold temperatures."""
    camera_optimizer_mode: Literal["off", "SO3xR3", "SE3"] = "SO3xR3"
    """Pose optimization strategy to use. Recommended to be SO3xR3."""


class ThermalNerfactoModel(NerfactoModel):

    config: ThermalNerfactoModelConfig

    def __init__(
        self,
        config: ThermalNerfactoModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        config.camera_optimizer = CameraOptimizerConfig(
            mode=config.camera_optimizer_mode
        )
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.config = config
        self.max_temperature = config.max_temperature
        self.min_temperature = config.min_temperature

    def get_image_metrics_and_images(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        threshold: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(
            outputs=outputs, batch=batch
        )

        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs[
            "rgb"
        ]  # Blended with background (black if random background)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        mae_foreground = mae_thermal(
            gt_rgb,
            predicted_rgb,
            self.config.cold,
            self.max_temperature,
            self.min_temperature,
            threshold=threshold,
        )
        mae = mae_thermal(
            gt_rgb,
            predicted_rgb,
            self.config.cold,
            self.max_temperature,
            self.min_temperature,
            threshold=None,
        )
        metrics_dict["mae_foreground"] = float(mae_foreground.item())
        metrics_dict["mae"] = float(mae.item())
        return metrics_dict, images_dict
