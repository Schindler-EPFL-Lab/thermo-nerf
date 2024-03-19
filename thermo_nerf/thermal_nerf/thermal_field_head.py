from enum import Enum
from typing import Callable, Optional, Union

from jaxtyping import Shaped
from nerfstudio.field_components.base_field_component import FieldComponent
from torch import Tensor, nn


class FieldHeadNamesT(Enum):
    """Possible field outputs"""

    THERMAL = "thermal"


class BaseThermalFieldHead(FieldComponent):
    """Base field output

    Args:
        out_dim: output dimension for renderer
        field_head_name: Field type
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(
        self,
        out_dim: int,
        field_head_name: FieldHeadNamesT,
        in_dim: Optional[int] = None,
        activation: Optional[Union[nn.Module, Callable]] = None,
    ) -> None:
        """`out_dim` represents the output dimension for the renderer.
        `field_head_name` is the type of field output.
        `in_dim` is the input dimension. If not defined in the constructor, it must be
        set later. `activation` is the output head activation."""
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.field_head_name = field_head_name
        self.net = None
        if in_dim is not None:
            self.in_dim = in_dim
            self._construct_net()

    def set_in_dim(self, in_dim: int) -> None:
        """Set input dimension of Field Head"""
        self.in_dim = in_dim
        self._construct_net()

    def _construct_net(self):
        self.net = nn.Linear(self.in_dim, self.out_dim)

    def forward(
        self, in_tensor: Shaped[Tensor, "*bs in_dim"]
    ) -> Shaped[Tensor, "*bs out_dim"]:
        """
        Process network output for renderer

        `in_tensor` is the network input.

        :return: Render head output
        """
        if not self.net:
            raise SystemError(
                "in_dim not set. Must be provided to constructor, or set_in_dim()"
                " should be called."
            )
        out_tensor = self.net(in_tensor)
        if self.activation:
            out_tensor = self.activation(out_tensor)
        return out_tensor
