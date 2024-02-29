# ThermoNeRF

This repo shows the main package used for ThermoNeRF.
The ThermoNeRF package is built on top of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). 
A more complete repo will published upon acceptance. 


## Setup
Clone the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
repo into this repo and follow the instalment instructions on the official Nerfstduio repo.


## Train and Evaluate 
To train and evaluate ThermoNeRF, first download our dataset and then use the following scripts 

```
python scripts/train_eval_script.py --data DATA_PATH --model_type thermal-nerf --max_num_iterations ITERATIONS 
```

## Evaluate 
To evaluate a model, run the following script.

```
python scripts/eval_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --output_folder RESULTS_PATH
```

## Render
Rendering can be done by indeifying specific camera poses (camera path) and query them from your trained model. 
For more infromation about it, check the [Nerfstudio Documentation](https://docs.nerf.studio/quickstart/viewer_quickstart.html)

To render a path of a scpefic scene using a pretrained model, use the following script
```
python scripts/render_video_script.py --dataset_path DATA_PATH --model_uri MODEL_PATH --camera_path_filename CAMERA_PATH_JSON --output_dir RENDER_RESULTS_PATH
```


## Model
We will be releasing our pretrained models on our scenes. 