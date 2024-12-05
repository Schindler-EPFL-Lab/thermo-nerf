from pathlib import Path

import numpy as np
import numpy.typing as npt
from nerfstudio.data.datasets.base_dataset import InputDataset
from PIL import Image


class ConcatDataset(InputDataset):
    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """
        Obtain numpy images with thermal information for concat-nerf

        Use `image_idx` to obtain original images and return the image of
        shape (H, W, 3 or 4).

        :returns: numpy image with thermal inforamtion
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename).convert("RGB")

        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)

        thermal_path = Path(image_filename.parent.parent, "thermal")
        thermal_images = [f for f in thermal_path.glob(image_filename.stem+".*")]

        if len(thermal_images) > 1:
            raise RuntimeError(
                "To many thermal file corresponding to ",
                thermal_path.name,
                ". Corresponding files:",
                str(thermal_images),
            )

        thermal_image_path = thermal_images[0]
        thermal_image = Image.open(thermal_image_path).convert("L")
        thermal_image = np.array(thermal_image, dtype="uint8")
        image = np.concatenate([image, thermal_image[..., None]], axis=-1)

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image
