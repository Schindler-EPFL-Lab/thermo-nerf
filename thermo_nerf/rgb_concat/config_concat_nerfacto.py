from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig

from thermo_nerf.nerfstudio_config.pipeline_tracking import (
    VanillaPipelineTrackingConfig,
)
from thermo_nerf.rgb_concat.concat_dataset import ConcatDataset
from thermo_nerf.rgb_concat.concat_nerfacto_model import ConcatNerfModelConfig

concat_nerf_config = TrainerConfig(
    method_name="concat_nerf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineTrackingConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[ConcatDataset],
            dataparser=NerfstudioDataParserConfig(eval_mode="filename"),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=ConcatNerfModelConfig(
            eval_num_rays_per_chunk=1 << 16,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0001, max_steps=200000
            ),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0001, max_steps=200000
            ),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
    vis="viewer",
)
