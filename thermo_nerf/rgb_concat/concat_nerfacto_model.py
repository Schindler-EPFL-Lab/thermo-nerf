from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.utils import colormaps
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thermo_nerf.nerfacto_config.thermal_nerfacto import (
    ThermalNerfactoModel,
    ThermalNerfactoModelConfig,
)
from thermo_nerf.rendered_image_modalities import RenderedImageModality
from thermo_nerf.rgb_concat.concat_field import ConcatNerfactoTField
from thermo_nerf.rgb_concat.rgbt_renderer import RGBTRenderer
from thermo_nerf.thermal_nerf.thermal_metrics import mae_thermal


@dataclass
class ConcatNerfModelConfig(ThermalNerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ConcatNerfModel)
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    thermal_loss_weight: float = 1.0
    """optimisation weight of the thermal loss."""
    pass_thermal_gradients: bool = True
    """Whether to pass thermal gradients."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use"""
    max_temperature: float = 1.0
    """Maximum temperature in the dataset."""
    min_temperature: float = 0.0
    """Minimum temperature in the dataset."""
    threshold: float = 0.0
    """Threshold for the thermal images that separated foreground from background."""
    cold: bool = False
    """Flag to indicate if the dataset includes cold temperatures."""


class ConcatNerfModel(ThermalNerfactoModel):
    """
    ConcatNerfModel extends NerfactoModel to support
    concatenated rgb and thermal images
    """

    config: ConcatNerfModelConfig

    def __init__(
        self,
        config: ConcatNerfModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.config = config
        self.max_temperature = config.max_temperature
        self.min_temperature = config.min_temperature

    def populate_modules(self):
        """Set the fields and modules."""

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = ConcatNerfactoTField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=(
                self.config.use_average_appearance_embedding
            ),
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(
                    step,
                    [0, self.config.proposal_warmup],
                    [0, self.config.proposal_update_every],
                ),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = (
            None  # None is for piecewise as default (see ProposalNetworkSampler)
        )
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # renderers
        self.renderer_rgb = RGBTRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs[RenderedImageModality.ACCUMULATION.value],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.training:
            loss_dict["interlevel_loss"] = (
                self.config.interlevel_loss_mult
                * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = (
                self.config.distortion_loss_mult * metrics_dict["distortion"]
            )
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = (
                    self.config.orientation_loss_mult
                    * torch.mean(outputs["rendered_orientation_loss"])
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = (
                    self.config.pred_normal_loss_mult
                    * torch.mean(outputs["rendered_pred_normal_loss"])
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image

        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_image_metrics_and_images(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        threshold: float | None = None,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs[
            "rgb"
        ]  # Blended with background (black if random background)

        acc = colormaps.apply_colormap(
            outputs[RenderedImageModality.ACCUMULATION.value]
        )
        depth = colormaps.apply_depth_colormap(
            outputs[RenderedImageModality.DEPTH.value],
            accumulation=outputs[RenderedImageModality.ACCUMULATION.value],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        gt_rgb = gt_rgb[:, 3, :, :].unsqueeze(dim=0)
        predicted_rgb = predicted_rgb[:, 3, :, :].unsqueeze(dim=0)

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
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

        gt_rgb = torch.repeat_interleave(gt_rgb, 3, dim=1)
        predicted_rgb = torch.repeat_interleave(predicted_rgb, 3, dim=1)

        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        metrics_dict["mae_thermal_foreground"] = float(mae_foreground.item())
        metrics_dict["mae_thermal"] = float(mae.item())

        images_dict = {
            RenderedImageModality.RGB.value: combined_rgb,
            RenderedImageModality.ACCUMULATION.value: combined_acc,
            RenderedImageModality.DEPTH.value: combined_depth,
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs[RenderedImageModality.ACCUMULATION.value],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
