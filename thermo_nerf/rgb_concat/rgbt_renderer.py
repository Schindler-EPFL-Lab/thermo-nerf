import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor

from thermo_nerf.base_renderer import BACKGROUNDCOLOR, BaseRenderer


class RGBTRenderer(BaseRenderer):
    """
    Renders RGBT images.
    """

    @staticmethod
    def combine(
        inputs: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BACKGROUNDCOLOR = "random",
        ray_indices: Int[Tensor, "num_samples"] | None = None,
        num_rays: int | None = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image.
        If background color is random, no BG color is added - as if the background
        was black!

        `input` and `weights` are the rgb and weights for each sample.
        `background_color` is the background color as thermal. When sampes are packed,
        `ray_indices` is used as the ray index for each sample, and `num_rays` is the
        number of rays.

        :return: Outputs rgb values.
        """
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError(
                    "Background color 'last_sample' not implemented for packed samples."
                )
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=inputs, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * inputs, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        rgb_only = comp_rgb[..., :3]
        if background_color == "random":
            # If background color is random, the predicted color is returned without
            # blending, as if the background color was black.
            return comp_rgb

        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = inputs[..., -1, :]
        else:
            background_color = RGBTRenderer.get_background_color(
                background_color, shape=rgb_only.shape, device=comp_rgb.device
            )
        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color * (1.0 - accumulated_weight)
        return comp_rgb

    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        pred_accumulation: Tensor,
        gt_image: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for loss
        computation.

        `gt_image` is the ground truth image and `pred_image` is the predicted RGB
        values (without background blending). `pred_accumulation` is the predicted
        opacity/accumulation.

        :return: A tuple of the predicted and ground truth RGB values.

        """

        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)

        return pred_image, gt_image
