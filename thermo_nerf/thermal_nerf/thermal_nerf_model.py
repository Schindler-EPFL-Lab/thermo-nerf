from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    interlevel_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.utils import colormaps
from torch import Tensor, nn
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from thermo_nerf.nerfacto_config.thermal_nerfacto import (
    ThermalNerfactoModel,
    ThermalNerfactoModelConfig,
)
from thermo_nerf.rendered_image_modalities import RenderedImageModality
from thermo_nerf.thermal_nerf.thermal_field import ThermalNerfactoTField
from thermo_nerf.thermal_nerf.thermal_field_head import FieldHeadNamesT
from thermo_nerf.thermal_nerf.thermal_metrics import mae_thermal
from thermo_nerf.thermal_nerf.thermal_renderer import ThermalRenderer


@dataclass
class ThermalNerfModelConfig(ThermalNerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ThermalNerfModel)
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    thermal_loss_weight: float = 1.0
    """optimisation weight of the thermal loss."""
    pass_thermal_gradients: bool = True
    """Whether to pass thermal gradients."""


class ThermalNerfModel(ThermalNerfactoModel):
    """
    ThermalNerfModel extends NerfactoModel to support thermal images
    as a separate modality from RGB images.
    """

    config: ThermalNerfModelConfig

    def __init__(
        self,
        config: ThermalNerfModelConfig,
        metadata: dict,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        if RenderedImageModality.THERMAL.value not in metadata.keys():
            raise ValueError("Thermal images not found in metadata.")

        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.config = config
        self.max_temperature = config.max_temperature
        self.min_temperature = config.min_temperature

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def populate_modules(self):
        """
        Initialises the fields and renderers.
        """

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = ThermalNerfactoTField(
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
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,  # noqa: E501
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
            use_transient_embedding=self.config.use_transient_embedding,
            pass_thermal_gradients=self.config.pass_thermal_gradients,
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
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        # renderer
        self.thermal_renderer = ThermalRenderer()
        # losses
        self.thermal_loss = MSELoss()

    def get_outputs(self, ray_bundle: RayBundle):
        """
        Runs forward pass of the model and calculates the outputs.

        :returns: Dictionary containing the required outputs of
        the model.
        """

        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_outputs = self.field.forward(
            ray_samples, compute_normals=self.config.predict_normals
        )
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(
                field_outputs, ray_samples
            )

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(
            weights=weights, ray_samples=ray_samples
        )
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            RenderedImageModality.ACCUMULATION.value: accumulation,
            RenderedImageModality.DEPTH.value: depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(
                normals=field_outputs[FieldHeadNames.NORMALS], weights=weights
            )
            pred_normals = self.renderer_normals(
                field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights
            )
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )
        thermal = self.thermal_renderer(field_outputs[FieldHeadNamesT.THERMAL], weights)

        outputs[RenderedImageModality.THERMAL.value] = thermal

        return outputs

    def get_loss_dict(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        metrics_dict=None,
    ):
        """
        Calculates all the losses for the model 'outputs' and the ground truths
        defined in 'batch'.
        """
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs[RenderedImageModality.ACCUMULATION.value],
            gt_image=image,
        )

        if self.field.pass_rgb_gradients:
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

        thermal_batch = batch[RenderedImageModality.THERMAL.value].to(self.device)

        if self.field.pass_thermal_gradients:
            loss_dict[RenderedImageModality.THERMAL.value] = self.thermal_loss(
                outputs[RenderedImageModality.THERMAL.value], thermal_batch
            )

        return loss_dict

    def get_image_metrics_and_images(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        threshold: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Tensor]]:
        """
        Outputs a dictionary of metrics and images for rendering and viewing.

        :returns: two dicts one for the metrics and and another for the images.
        """
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        thermal = colormaps.apply_float_colormap(
            outputs[RenderedImageModality.THERMAL.value], colormap="gray"
        )
        gt_thermal = colormaps.apply_float_colormap(
            batch[RenderedImageModality.THERMAL.value].to(self.device), colormap="gray"
        )
        combined_thermal = torch.cat([gt_thermal, thermal], dim=1)

        images_dict.update({RenderedImageModality.THERMAL.value: thermal})
        images_dict.update(
            {RenderedImageModality.THERMAL_COMBINED.value: combined_thermal}
        )

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_thermal = torch.moveaxis(batch[RenderedImageModality.THERMAL.value], -1, 0)[
            None, ...
        ]
        predicted_thermal = torch.moveaxis(
            outputs[RenderedImageModality.THERMAL.value], -1, 0
        )[None, ...]

        psnr = self.psnr(gt_thermal, predicted_thermal)
        ssim = self.ssim(gt_thermal, predicted_thermal)
        mae_foreground = mae_thermal(
            gt_thermal,
            predicted_thermal,
            self.config.cold,
            self.max_temperature,
            self.min_temperature,
            threshold=threshold,
        )
        mae = mae_thermal(
            gt_thermal,
            predicted_thermal,
            self.config.cold,
            self.max_temperature,
            self.min_temperature,
            threshold=None,
        )

        # repeat channels for lpips
        gt_thermal = torch.repeat_interleave(gt_thermal, 3, dim=1)
        predicted_thermal = torch.repeat_interleave(predicted_thermal, 3, dim=1)
        lpips = self.lpips(gt_thermal, predicted_thermal)

        metrics_dict["psnr_thermal"] = float(psnr.item())
        metrics_dict["ssim_thermal"] = float(ssim)

        metrics_dict["lpips_thermal"] = float(lpips)
        metrics_dict["mae_thermal_foreground"] = float(mae_foreground.item())
        metrics_dict["mae_thermal"] = float(mae.item())

        return metrics_dict, images_dict
