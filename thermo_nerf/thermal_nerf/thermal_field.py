from typing import Literal

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.fields.nerfacto_field import NerfactoField
from torch import Tensor, nn

from thermo_nerf.thermal_nerf.thermal_field_head import (
    BaseThermalFieldHead,
    FieldHeadNamesT,
)


class ThermalFieldHead(BaseThermalFieldHead):
    """Thermal output"""

    def __init__(self, in_dim: int | None = None) -> None:
        """`in_dim` is the input dimension. If not defined in the constructor,
        it must be set later.
        """
        super().__init__(
            in_dim=in_dim,
            out_dim=1,
            field_head_name=FieldHeadNamesT.THERMAL,
            activation=None,
        )


class ThermalNerfactoTField(NerfactoField):
    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion | None = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        pass_thermal_gradients: bool = False,
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers,
            hidden_dim,
            geo_feat_dim,
            num_levels,
            base_res,
            max_res,
            log2_hashmap_size,
            num_layers_color,
            num_layers_transient,
            features_per_level,
            hidden_dim_color,
            hidden_dim_transient,
            appearance_embedding_dim,
            transient_embedding_dim,
            use_transient_embedding,
            use_semantics,
            num_semantic_classes,
            pass_semantic_gradients,
            use_pred_normals,
            use_average_appearance_embedding,
            spatial_distortion,
            1.0,
            implementation,
        )

        self.mlp_thermal = MLP(
            in_dim=self.geo_feat_dim,
            num_layers=2,
            layer_width=64,
            out_dim=hidden_dim_transient,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.field_head_thermal = ThermalFieldHead(
            in_dim=self.mlp_thermal.get_out_dim()
        )
        self.pass_thermal_gradients = pass_thermal_gradients
        self.training_iteration = 0

        self.pass_rgb_gradients = True

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Tensor | None = None
    ) -> dict[FieldHeadNamesT | FieldHeadNames, Tensor]:
        assert density_embedding is not None

        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training and self.embedding_appearance is not None:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            assert self.embedding_appearance is not None
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = (
                self.mlp_transient(transient_input)
                .view(*outputs_shape, -1)
                .to(directions)
            )

            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = (
                self.field_head_transient_density(x)
            )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        thermal_input = density_embedding.view(-1, self.geo_feat_dim)
        if not self.pass_thermal_gradients:
            thermal_input = thermal_input.detach()

        mlp_output = (
            self.mlp_thermal(thermal_input).view(*outputs_shape, -1).to(directions)
        )
        thermal = self.field_head_thermal(mlp_output)
        outputs.update({FieldHeadNamesT.THERMAL: thermal})

        return outputs

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False
    ) -> dict[FieldHeadNamesT | FieldHeadNames, Tensor]:
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding
        )
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
