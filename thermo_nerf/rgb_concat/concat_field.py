from typing import Literal

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField
from torch import Tensor, nn


class ConcatNerfactoTField(NerfactoField):
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

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim()
            + self.geo_feat_dim
            + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=4,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )
