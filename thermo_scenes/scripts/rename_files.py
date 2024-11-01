from pathlib import Path
from typing import List, Tuple

import tyro
from nerfstudio.process_data.process_data_utils import list_images
from PIL import Image as PILImage
from PIL.Image import Image


def load_images(
    base_path: Path,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Loads images from subfolders of `base_path`

    Loads rgb and thermla images in paired from the `base_path` for
    both test and train sets.

    :returns:   a list of paired thermal and rgb train images and
                a list of paired thermal and rgb test images
    """
    rgb_train_dir = Path(base_path, "rgb_train")
    thermal_train_dir = Path(base_path, "thermal_train")
    rgb_eval_dir = Path(base_path, "rgb_eval")
    thermal_eval_dir = Path(base_path, "thermal_eval")

    directories = [rgb_train_dir, thermal_train_dir, rgb_eval_dir, thermal_eval_dir]

    for dir_path in directories:
        if not dir_path.exists():
            raise Exception(f"Error: Directory {dir_path} not found in {base_path}")

    rgb_train_images = list_images(rgb_train_dir)
    thermal_train_images = list_images(thermal_train_dir)
    rgb_eval_images = list_images(rgb_eval_dir)
    thermal_eval_images = list_images(thermal_eval_dir)

    if len(rgb_train_images) != len(thermal_train_images):
        raise Exception(
            "Error: The number of RGB and thermal train images do not match."
        )

    if len(rgb_eval_images) != len(thermal_eval_images):
        raise Exception(
            "Error: The number of RGB and thermal eval images do not match."
        )

    train_images = list(zip(sorted(rgb_train_images), sorted(thermal_train_images)))
    eval_images = list(zip(sorted(rgb_eval_images), sorted(thermal_eval_images)))

    return train_images, eval_images


def resize_image(image_path: Path, target_size: tuple) -> Image:
    """
    Resizes the image of the `image_path` to the `target_size`

    :returns: resized Image
    """
    with PILImage.open(image_path) as img:
        resized_img = img.resize(target_size, PILImage.Resampling.LANCZOS)
        return resized_img


def save_images(
    images: list[tuple[Path, Path]],
    base_path: Path,
    dir_name_rgb: str,
    dir_name_thermal: str,
    prefix: str,
) -> None:
    """
    Resizes and saves images in the `images` list

    Resized rgb images saved to "`base_path`/`dir_name_rgb`" folder
    Resized thermal images saved to "`base_path`/`dir_name_thermal`" folder

    `prefix`: name of the image series
    """
    if not images:
        return

    Path(base_path, dir_name_rgb).mkdir(exist_ok=True)
    Path(base_path, dir_name_thermal).mkdir(exist_ok=True)

    for index, (rgb_path, thermal_path) in enumerate(images, start=1):
        rgb_filename = f"{prefix}_{index:05d}.png"
        thermal_filename = f"{prefix}_{index:05d}.png"

        with PILImage.open(thermal_path) as thermal_img:
            target_size = (thermal_img.width, thermal_img.height)

        resized_rgb = resize_image(rgb_path, target_size)
        resized_thermal = resize_image(thermal_path, target_size)

        resized_rgb.save(Path(base_path, dir_name_rgb, rgb_filename))
        resized_thermal.save(Path(base_path, dir_name_thermal, thermal_filename))


def process(path_to_folder: Path) -> None:
    """
    Main function: Loads, processes and saves images from `base_path`
    """
    train_images, eval_images = load_images(path_to_folder)
    save_images(
        train_images,
        path_to_folder,
        "rgb_train_processed",
        "thermal_train_processed",
        "frame_train",
    )
    save_images(
        eval_images,
        path_to_folder,
        "rgb_eval_processed",
        "thermal_eval_processed",
        "frame_eval",
    )


def main() -> None:
    tyro.cli(process)


if __name__ == "__main__":
    main()
