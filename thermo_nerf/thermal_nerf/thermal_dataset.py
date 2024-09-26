from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from thermo_nerf.rendered_image_modalities import RenderedImageModality


class ThermalDataset(InputDataset):
    """
    Dataset class that returns thermal images.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
        RenderedImageModality.THERMAL.value
    ]

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__(dataparser_outputs, scale_factor)
        assert RenderedImageModality.THERMAL.value in dataparser_outputs.metadata.keys()
        self.thermal_filenames = self.metadata[RenderedImageModality.THERMAL.value]
        self.kernel_size = kernel_size

    def get_metadata(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Loads the ground truths thermal images.

        :return: Dictionary containing the ground truth thermal data.
        """

        filepath = Path(self.thermal_filenames[data["image_idx"]])

        thermal_data = self.get_thermal_tensors_from_path(
            filepath=filepath,
            scale_factor=self.scale_factor,
        )

        return {RenderedImageModality.THERMAL.value: thermal_data}

    @staticmethod
    def get_thermal_tensors_from_path(
        filepath: Path, scale_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Reads thermal images from the given `filepath`

        :returns: A tensor representing the thermal image.
        """

        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"No file found at {filepath}")

        image = image / 255.0
        image = image.astype(np.float32)

        if scale_factor != 1.0:
            width, height = image.shape[1], image.shape[0]
            image = cv2.resize(
                image, (int(width * scale_factor), int(height * scale_factor))
            )

        return torch.from_numpy(image[:, :, np.newaxis])
