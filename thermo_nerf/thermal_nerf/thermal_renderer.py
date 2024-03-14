from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from thermo_nerf.base_renderer import BACKGROUNDCOLOR, BaseRenderer


class ThermalRenderer(BaseRenderer):
    """
    Renders thermal images the same way we render semantic labels
    and depth images.
    """

    def __init__(self, background_color: BACKGROUNDCOLOR = "random") -> None:
        super().__init__()
        self.background_color: BACKGROUNDCOLOR = background_color

    @staticmethod
    def combine(
        inputs: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BACKGROUNDCOLOR = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        If background color is random, no BG color is added - as if the background
        was black!

        `input` and `weights` are the thermal and weights for each sample.
        `background_color` is the background color as thermal. When sampes are packed,
        `ray_indices` is used as the ray index for each sample, and `num_rays` is the
        number of rays.

        :return: Outputs thermal values.
        """
        background_color = "last_sample"
        if ray_indices is not None and num_rays is not None:
            raise NotImplementedError(
                "Background color 'last_sample' not implemented for packed samples."
            )

        comp_thermal = torch.sum(weights * inputs, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)

        if inputs.shape[-1] == 1 or len(inputs.shape) == 2:
            # repeat channels
            thermal_3channels = inputs.repeat(1, 1, 3)
        else:
            thermal_3channels = inputs

        if background_color == "random":
            return comp_thermal

        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = inputs[..., -1, :]

        else:
            background_color = ThermalRenderer.get_background_color(
                background_color,
                shape=thermal_3channels.shape,
                device=comp_thermal.device,
            )
        assert isinstance(background_color, torch.Tensor)
        comp_thermal = comp_thermal + background_color * (1.0 - accumulated_weight)
        return comp_thermal
