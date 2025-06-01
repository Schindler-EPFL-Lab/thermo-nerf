# ThermoNeRF

Build RGB+Thermal models easily!

This repo is the official Pytorch code for [ThermoNerf](https://www.sciencedirect.com/science/article/abs/pii/S1474034625002381). PDF [here](https://arxiv.org/abs/2403.12154).

![thermal EPFL pavillon](images/epfl_pavillon.gif)
![thermal cup](images/cup.gif)

## Introduction

![Summary of the method](images/summary.png)

We present ThermoNeRF, a novel multimodal approach based on Neural Radiance Fields, capable of rendering new RGB and thermal views of a scene jointly.
To overcome the lack of texture in thermal images, we use paired RGB and thermal images to learn scene density, while distinct networks estimate color and temperature information.

One of the unique contribution of ThermoNeRF is that both RGB and thermal data are used to backpropagate to the density, leading to a consistent representation of the scene for both RGB and thermal.

The ThermoNeRF package is built on top of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

## Dataset

We introduce [ThermoScenes](https://zenodo.org/records/10835108?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhlOWI4MTVmLWZlOGUtNDA0Mi1hMWE1LWM5OWYwODE1MjNkNSIsImRhdGEiOnt9LCJyYW5kb20iOiI3NDUwNzM3ZjAxNTlkZWVjNzI1NWY0MmYyMTQxMzdkMyJ9.3Ga9svyICCtX8FwVOWx0NSCx8AHzjb-aqbO1VRLVfUf_CK6fp7sPz2WopezuH3iPxrTag7ivoG1p56ND1eNpVg), a new dataset to palliate the lack of available RGB+thermal datasets for scene reconstruction.

We also [provide instructions](thermo_scenes/docs/Collect_new_dataset.md) on how to build your own thermal scene.
If you create a new scene and wish to share it with everyone, send it to us and we may add it to ThermoScenes.

## Installation

ThermoNeRF was tested under Python 3.10 and with torch `1.13.1`.

Clone the package.
If you want to use FLIR cameras and build your dataset, clone with the submodule:

```bash
$ git clone --recurse-submodules git@github.com:Schindler-EPFL-Lab/thermo-nerf.git
```

Install the package by running `uv sync --all-extras --prerelease=allow` and then thermoNeRF should be ready to go.

## Train and Evaluate

To train and evaluate ThermoNeRF, first download our dataset and then use the following scripts

```bash
python thermo_nerf/scripts/train_eval_script.py --data-asset-path DATA_PATH --model-type thermal-nerf --max-num-iterations ITERATIONS
```

E.g.

```bash
python thermo_nerf/scripts/train_eval_script.py --data data/ThermoScenes/double_robot/ --model_type thermal-nerf --max_num_iterations 1000
```

## Evaluate

To evaluate a model, run the following script.

```bash
python thermo_nerf/scripts/eval_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --output_folder RESULTS_PATH
```

## Render

Rendering can be done by creating specific camera poses (camera path) and query them from your trained model.
For more information about it, check [Nerfstudio Documentation](https://docs.nerf.studio/quickstart/viewer_quickstart.html)

To render a path of a scpefic scene using a pretrained model, use the following script

```bash
python thermo_nerf/scripts/render_video_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --camera_path_filename CAMERA_PATH_JSON --output_dir RENDER_RESULTS_PATH
```
## Visualization

A more exhaustive evaluation of the model can be done using [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) viewer.
To render a thermal model run the related script:
```bash
python thermo_nerf/scripts/thermal_viewer.py --model-path MODEL_PATH --dataset-path DATA_PATH
```

## Contribute

We welcome contributions! Feel free to fork and submit PRs.

We format code using [ruff](https://docs.astral.sh/ruff) and follow PEP8.
The code needs to be type annotated and following our documentation style.

## How to cite

```bibtex
@article{hassan2025thermonerf,
  title={ThermoNeRF: A multimodal Neural Radiance Field for joint RGB-thermal novel view synthesis of building facades},
  author={Hassan, Mariam and Forest, Florent and Fink, Olga and Mielle, Malcolm},
  journal={Advanced Engineering Informatics},
  volume={65},
  pages={103345},
  year={2025},
  publisher={Elsevier}
}
```
