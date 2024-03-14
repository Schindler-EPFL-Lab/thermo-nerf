# ThermoNeRF

The ThermoNeRF package is built on top of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

The Nerfstudio commit hash used in our experiments: 3dd162aae8ae7c166928e1f02bf97e7febe3a18e

A more complete repo will published upon acceptance.

## Setup

ThermoNeRF was tested under Python 3.10 with torch 1.
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
For more infromation about it, check the [Nerfstudio Documentation](https://docs.nerf.studio/quickstart/viewer_quickstart.html)

To render a path of a scpefic scene using a pretrained model, use the following script

```bash
python scripts/render_video_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --camera_path_filename CAMERA_PATH_JSON --output_dir RENDER_RESULTS_PATH
```
