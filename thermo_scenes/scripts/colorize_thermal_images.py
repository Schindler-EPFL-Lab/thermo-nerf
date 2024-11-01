import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from PIL import Image


def save_image_with_colormap(
    image: np.ndarray,
    output_img: Path,
    min_temp,
    max_temp,
    display_cmap: bool = False,
    show: bool = False,
) -> None:
    """
    Save the `image` with a colormap to the `output_img` path.
    `min_temp` and `max_temp` are the minimum and maximum temperatures in the dataset.
    `display_cmap` is a boolean to display the colorbar on the plot and `show` is a
    boolean to display the plot live.
    """
    cmap = plt.cm.magma

    image = (image - min_temp) / (max_temp - min_temp)

    # Apply colormap to the normalized image
    colored_image = cmap(image)

    # Add colorbar
    plt.imshow(
        colored_image,
        cmap=cmap,
        vmin=min_temp,
        vmax=max_temp,
    )

    plt.axis("off")
    if display_cmap:
        cbar = plt.colorbar(orientation="vertical", fraction=0.05, cmap=cmap)
        cbar.set_label("Temperature", rotation=270, labelpad=15)
        cbar.ax.yaxis.label.set_fontsize(18)
        cbar.ax.tick_params(labelsize=18)
    plt.savefig(
        output_img,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    if show:
        plt.show()


def main(images: Path, output_dir: Path, temperatures_bound: Path) -> None:
    """
    Colorize the thermal images in `images` using the temperature bounds in
    `temperatures_bound` and save the colorized images in `output_dir`.
    """

    if temperatures_bound.suffix != ".json":
        raise ValueError("The temperature_bound file must be a json file.")

    if images == output_dir:
        raise ValueError("The images_path and output_dir cannot be the same.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images.iterdir():

        if not (
            image_path.suffix == ".jpg"
            or image_path.suffix == ".png"
            or image_path.suffix == ".PNG"
            or image_path.suffix == ".jpeg"
        ):
            continue
        reconstruction_img_path = Path(output_dir, image_path.name)

        with open(temperatures_bound, "r") as json_file:
            temperature_bounds = json.load(json_file)
        max_temp = temperature_bounds["absolute_max_temperature"]
        min_temp = temperature_bounds["absolute_min_temperature"]

        greyscale_img = np.array(Image.open(image_path).convert("L"))
        greyscale_img = greyscale_img / 255.0
        greyscale_img = greyscale_img * (max_temp - min_temp) + min_temp
        save_image_with_colormap(
            image=greyscale_img,
            output_img=reconstruction_img_path,
            min_temp=min_temp,
            max_temp=max_temp,
            display_cmap=False,
        )


if __name__ == "__main__":
    tyro.cli(main)
