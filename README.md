# ThermoNeRF

This repo is the official Pytorch code from [ThermoNerf](https://arxiv.org/abs/2403.12154).

## Introduction

![Summary of the method](images/summary.png)

we present ThermoNeRF, a novel multimodal approach based on Neural Radiance Fields, capable of rendering new RGB and thermal views of a scene jointly.
To overcome the lack of texture in thermal images, we use paired RGB and thermal images to learn scene density, while distinct networks estimate color and temperature information.

One of the unique contribution of ThermoNeRF is that both RGB and thermal data are used to backpropagate to the density, leading to a consistent representation of the scene for both RGB and thermal.

The ThermoNeRF package is built on top of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).
The Nerfstudio commit hash used in our experiments: [3dd162aae8ae7c166928e1f02bf97e7febe3a18e](https://github.com/nerfstudio-project/nerfstudio/tree/3dd162aae8ae7c166928e1f02bf97e7febe3a18e).

## Dataset

We introduce [ThermoScenes](https://zenodo.org/records/10835108?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhlOWI4MTVmLWZlOGUtNDA0Mi1hMWE1LWM5OWYwODE1MjNkNSIsImRhdGEiOnt9LCJyYW5kb20iOiI3NDUwNzM3ZjAxNTlkZWVjNzI1NWY0MmYyMTQxMzdkMyJ9.3Ga9svyICCtX8FwVOWx0NSCx8AHzjb-aqbO1VRLVfUf_CK6fp7sPz2WopezuH3iPxrTag7ivoG1p56ND1eNpVg), a new dataset to palliate the lack of available RGB+thermal datasets for scene reconstruction.

## Installation

ThermoNeRF was tested under Python 3.10 with torch `1.13.1`.
Install the package by running `pip install -e .` and then thermoNeRF should be ready to go.

## Train and Evaluate

To train and evaluate ThermoNeRF, first download our dataset and then use the following scripts

```bash
python scripts/train_eval_script.py --data-asset-path DATA_PATH --model-type thermal-nerf --max-num-iterations ITERATIONS
```

E.g.

```bash
python scripts/train_eval_script.py --data data/ThermoScenes/double_robot/ --model_type thermal-nerf --max_num_iterations 1000
```

## Evaluate

To evaluate a model, run the following script.

```bash
python scripts/eval_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --output_folder RESULTS_PATH
```

## Render

Rendering can be done by creating specific camera poses (camera path) and query them from your trained model.
For more information about it, check [Nerfstudio Documentation](https://docs.nerf.studio/quickstart/viewer_quickstart.html)

To render a path of a scpefic scene using a pretrained model, use the following script

```bash
python scripts/render_video_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --camera_path_filename CAMERA_PATH_JSON --output_dir RENDER_RESULTS_PATH
```

## Contribute

We welcome contributions! Feel free to fork and submit PRs.

We format code using [black](https://pypi.org/project/black/) and follow PEP8.
The code needs to be type annotated and following our documentation style.

## How to cite

For now the paper is on arxiv:

```bibtex
@misc{hassan2024thermonerf,
      title={ThermoNeRF: Multimodal Neural Radiance Fields for Thermal Novel View Synthesis},
      author={Mariam Hassan and Florent Forest and Olga Fink and Malcolm Mielle},
      year={2024},
      eprint={2403.12154},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
