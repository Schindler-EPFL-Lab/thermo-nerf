"""Datapaser for semantic-SDF/nerf formatted data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio,
    NerfstudioDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json

from thermo_nerf.rendered_image_modalities import RenderedImageModality


@dataclass
class ThermalDataParserConfig(NerfstudioDataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: Thermal)
    """Target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter
    to meter conversion."""
    downscale_factor: int = 1
    """Downscale image size"""
    scene_scale: float = 1.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the axis-aligned bbox will be scaled to this value.
    """
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""

    eval_mode: str = "filename"


@dataclass
class Thermal(Nerfstudio):
    """
    Thermal Dataparser

    The dataparser is responsible to prepare the data for the PyTorch dataset and
    dataloader.
    """

    config: ThermalDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """
        Reads the metadata, load the RGB images and their semantic
        segmentation, and store the camera poses.The `split` argument is a mandatory
        argument for all dataparsers. It is not used in our case.
        """

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        thermal_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))

            if "thermal_file_path" in frame:
                thermal_filepath = Path(frame["thermal_file_path"])
                thermal_fname = self._get_fname(
                    thermal_filepath,
                    data_dir,
                    downsample_folder_prefix="thermal_",
                )
                thermal_filenames.append(thermal_fname)

        has_split_files_spec = any(
            f"{split}_filenames" in meta for split in ("train", "val", "test")
        )
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(
                self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"]
            )
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were "
                    "not found: {unmatched_filenames}."
                )

            indices = [
                i for i, path in enumerate(image_filenames) if path in split_filenames
            ]

            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(
                f"The dataset's list of filenames for split {split} is missing."
            )
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(
                    image_filenames, self.config.train_split_fraction
                )
            elif self.config.eval_mode == "filename":
                print("Using filename split")
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(
                    image_filenames, self.config.eval_interval
                )
            elif self.config.eval_mode == "all":
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient
        # and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]

        thermal_filenames = (
            [thermal_filenames[i] for i in indices]
            if len(thermal_filenames) > 0
            else []
        )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = (
            float(meta["fl_x"])
            if fx_fixed
            else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        )
        fy = (
            float(meta["fl_y"])
            if fy_fixed
            else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        )
        cx = (
            float(meta["cx"])
            if cx_fixed
            else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        )
        cy = (
            float(meta["cy"])
            if cy_fixed
            else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        )
        height = (
            int(meta["h"])
            if height_fixed
            else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        )
        width = (
            int(meta["w"])
            if width_fixed
            else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        )
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                RenderedImageModality.THERMAL.value: (
                    thermal_filenames if len(thermal_filenames) > 0 else None
                ),
            },
        )
        return dataparser_outputs
