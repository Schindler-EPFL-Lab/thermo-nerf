from abc import abstractmethod
from typing import Literal, TypeAlias, Union

import torch
from jaxtyping import Float, Int
from nerfstudio.utils import colors
from torch import Tensor, nn

BACKGROUNDCOLOR: TypeAlias = Union[
    Literal["random", "last_sample", "black", "white"],
    Float[Tensor, "3"],
    Float[Tensor, "*bs 3"],
]


class BaseRenderer(nn.Module):
    """
    Renders thermal images the same way we render semantic labels
    and depth images.
    """

    BACKGROUND_COLOR_OVERRIDE: Float[Tensor, "3"] | None = None

    def __init__(self, background_color: BACKGROUNDCOLOR = "random") -> None:
        super().__init__()
        self.background_color: BACKGROUNDCOLOR = background_color

    @staticmethod
    def get_background_color(
        background_color: BACKGROUNDCOLOR,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> Float[Tensor, "3"] | Float[Tensor, "*bs 3"]:
        """Returns the RGB background color for a specified background color.

        `background_color` is the background color specification.
        Note that this function CANNOT be called for background_color being either
        "last_sample" or "random".

        :return: Background color as three-channel RGB.
        """

        assert background_color not in {"last_sample", "random"}
        assert shape[-1] == 3, "Background color must be RGB."
        if BaseRenderer.BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BaseRenderer.BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color]
        assert isinstance(background_color, Tensor)

        # Ensure correct shape
        return background_color.expand(shape).to(device)

    def forward(
        self,
        output: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Int[Tensor, "num_samples"] | None = None,
        num_rays: int | None = None,
        background_color: BACKGROUNDCOLOR | None = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        `output` and `weights` are the output values and weights for each sample. When
        samples are packed, `ray_indices` is the ray index for each sample,  and
        `num_rays` is the number of rays. `background_color` is the background color to
        use for rendering.

        :return: Outputs of `output` values.
        """

        if background_color is None:
            background_color = self.background_color

        if not self.training:
            output = torch.nan_to_num(output)
        output = self.combine(
            output,
            weights,
            background_color=background_color,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        if not self.training:
            torch.clamp_(output, min=0.0, max=1.0)

        return output

    @staticmethod
    @abstractmethod
    def combine(
        inputs: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BACKGROUNDCOLOR = "random",
        ray_indices: Int[Tensor, "num_samples"] | None = None,
        num_rays: int | None = None,
    ) -> Float[Tensor, "*bs 3"]:
        raise NotImplementedError("not implamented")
