# Thermal Images

- [Thermal Images](#thermal-images)
  - [Introduction](#introduction)
  - [Instructions](#instructions)

## Introduction

To process thermal images of Flir One camera, we mainly use the [Flir Extractor Repo](https://github.com/ITVRoC/FlirImageExtractor). It extracts raw information of the images and outputs raw thermal images, RGB images, and CSV files with the exact temperature value at every pixel.

We build on top of that to extract our own temperature images that directly encode the temperature values in greyscale images. To do that, we normalise the images based on the maximum and minimum temperatures in the entire dataset.

We also provide a visualiser for the greyscale images that shows the temperature value of every pixel once it's hovered on with the mouse.

## Instructions

To get your normalised thermal images, first create a folder and insert all your images from the Flir One app in it.

Then, clone the repo in your root directory as follows:

```
 git clone https://github.com/ITVRoC/FlirImageExtractor.git
```

Make sure to install the following packages:

```
sudo apt update
sudo apt install exiftool python-setuptools
sudo pip install numpy matplotlib pillow
```

To create the CSV files, rgb images, and the greyscale raw temperature images, run the following.

```
python scripts/process_thermal/preprocess_thermal.py --path-to-thermal-images <path_to_thermal> --path-to-thermal-images-curated <path_to_output_thermal> --path-to-rgb <path_to_rgb> --path-to-csv-files  <path_to_csv>
```

To visualise the temperature values on the images, run the following script.

```
python scripts/process_thermal/visualise_temperature.py --path_to_thermal_image <path_to_thermal_image> --path_to_txt_file <path_to_txt_file>
```
