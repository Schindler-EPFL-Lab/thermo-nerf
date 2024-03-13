from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import mlflow
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler


@dataclass
class VanillaPipelineTrackingConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipelineTracking)
    """target class to instantiate"""
    metrics_logging_freqency: float = 0.0075
    """Frequency at which the metrics are logged. [step^-1] """


class VanillaPipelineTracking(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineTrackingConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ) -> None:
        """The pipeline class for the vanilla nerf setup of multiple cameras for one or
        a few scene ; with metrics tracking at a given frequency.

        `config` is the configuration to instantiate the pipeline. Model and data will
        be located in `device`. Specify the number of machines available in `world_size`
        and the rank of the current machine with `local_rank`. A gradient scaler
        (`grad_scaler`) is used during training. Finally, specify the `test_mode`:
                - 'val': loads train/val datasets into memory
                - 'test': loads train/test dataset into memory
                - 'inference': does not load any dataset into memory

        This class can use two attributes. The `datamanager` and the `model` that will
        be used.
        """
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        self.counter = 0

    def get_train_loss_dict(
        self, step: int
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:

        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)
        if self.counter > int(1 / self.config.metrics_logging_freqency):
            mlflow.log_metrics(metrics_dict)
            mlflow.log_metrics(loss_dict)
            self.counter = 0
        else:
            self.counter += 1

        return model_outputs, loss_dict, metrics_dict
