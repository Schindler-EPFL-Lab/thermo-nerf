from typing import Literal, Optional, Union

import torch
from jaxtyping import Float, Int
from nerfstudio.utils import colors
from torch import Tensor, nn

BackgroundColor = Union[
    Literal["random", "last_sample", "black", "white"],
    Float[Tensor, "3"],
    Float[Tensor, "*bs 3"],
]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


class ThermalRenderer(nn.Module):
    """
    Renders thermal images the same way we render semantic labels
    and depth images.
    """

    def __init__(self, background_color: BackgroundColor = "random") -> None:
        super().__init__()
        self.background_color: BackgroundColor = background_color

    @classmethod
    def combine_thermal(
        cls,
        thermal: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        If background color is random, no BG color is added - as if the background was
        black!

        Args:
            thermal: thermal for each sample
            weights: Weights for each sample
            background_color: Background color as thermal.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs thermal values.
        """
        background_color = "last_sample"
        if ray_indices is not None and num_rays is not None:
            raise NotImplementedError(
                "Background color 'last_sample' not implemented for packed samples."
            )

        comp_thermal = torch.sum(weights * thermal, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)

        if thermal.shape[-1] == 1 or len(thermal.shape) == 2:
            # repeat channels
            thermal_3channels = thermal.repeat(1, 1, 3)
        else:
            thermal_3channels = thermal

        if background_color == "random":

            return comp_thermal

        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = thermal[..., -1, :]

        else:
            background_color = cls.get_background_color(
                background_color,
                shape=thermal_3channels.shape,
                device=comp_thermal.device,
            )
        assert isinstance(background_color, torch.Tensor)
        comp_thermal = comp_thermal + background_color * (1.0 - accumulated_weight)
        return comp_thermal

    @classmethod
    def get_background_color(
        cls,
        background_color: BackgroundColor,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        """Returns the thermal background color for a specified background color.

        Note:
            This function CANNOT be called for background_color being either
            "last_sample" or "random".

        Args:
            thermal: thermal for each sample.
            background_color: The background color specification.

        Returns:
            Background color as thermal.
        """
        assert background_color not in {"last_sample", "random"}
        assert shape[-1] == 3, "Background color must be thermal."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color]
        assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)

    def forward(
        self,
        thermal: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            thermal: thermal for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            background_color: The background color to use for rendering.

        Returns:
            Outputs of thermal values.
        """
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            thermal = torch.nan_to_num(thermal)
        thermal = self.combine_thermal(
            thermal,
            weights,
            background_color=background_color,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        if not self.training:
            torch.clamp_(thermal, min=0.0, max=1.0)

        return thermal
