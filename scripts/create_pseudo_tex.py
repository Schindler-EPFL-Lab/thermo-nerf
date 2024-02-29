import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mlflow
import tyro
from PIL import Image

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed

sys.path.append(".")
sys.path.append("./nerfstudio")


from rebel_nerf.pseudo_tex.image_segmentation import ImageSegmentation  # noqa: E402
from rebel_nerf.pseudo_tex.pseudo_tex import PseudoTeX  # noqa: E402


@dataclass
class Parameters:
    input_data_file: str
    """Path to the input data file."""
    output_file: str
    """Path to the output file."""
    output_size: list[int] = field(default_factory=lambda: [480, 640])
    """Desired output image size in the format 'width height'. Default is 800,600."""
    alpha: float = 0.5
    """Transpareny value for overlay."""


def main() -> None:
    """
    processes thermal and rgb images to create pseudo-TeX images.
    """

    parameters = tyro.cli(Parameters)

    thermal_data_folder = Path(parameters.input_data_file, "thermal")

    saving_folder = parameters.output_file
    width, height = parameters.output_size
    alpha = parameters.alpha

    for thermal_img in thermal_data_folder.iterdir():
        file_name = thermal_img.name
        rgb_filename = Path(parameters.input_data_file, "rgb", file_name)

        thermal_image = Image.open(thermal_img)

        rgb_image = cv2.imread(str(rgb_filename))

        semantic_segmentation = ImageSegmentation(rgb_image)
        semantic_image = semantic_segmentation.get_segmentation()

        pseudo_tex = PseudoTeX(thermal_image, semantic_image, width, height, alpha)

        pseudo_tex_image = pseudo_tex.pseudo_tex

        mlflow.log_image(pseudo_tex_image, file_name.split(".")[0] + ".png")

        cv2.imwrite(
            str(
                Path(
                    str(saving_folder),
                    str(file_name.split(".")[0]) + ".png",
                )
            ),
            pseudo_tex_image,
        )
        segmentation_folder = Path("outputs/segmentation")

        if not segmentation_folder.exists():
            segmentation_folder.mkdir(parents=True, exist_ok=True)

        texture_folder = Path("outputs/texture")

        if not texture_folder.exists():
            texture_folder.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(
                Path(
                    str(segmentation_folder),
                    str(file_name.split(".")[0]) + ".png",
                )
            ),
            pseudo_tex._semantic_image,
        )
        cv2.imwrite(
            str(
                Path(
                    str(texture_folder),
                    str(file_name.split(".")[0]) + ".png",
                )
            ),
            pseudo_tex._texture_image,
        )


if __name__ == "__main__":
    main()
