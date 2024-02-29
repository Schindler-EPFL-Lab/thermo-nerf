import random
import sys
from dataclasses import dataclass

import mlflow
import torch
import torchvision.transforms as transforms
import tyro
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
from rebel_nerf.image_translator.config.parameters import (  # noqa: E402
    TranslatorParameters,
)
from rebel_nerf.image_translator.data import dataset  # noqa: E402
from rebel_nerf.image_translator.model.translator import Translator  # noqa: E402


@dataclass
class Param(TranslatorParameters):
    dataset_name: str
    """Only used in naming the model to register."""


def train():
    mlflow.start_run()
    client = MlflowClient()
    training_args = tyro.cli(Param)
    mlflow.log_params(training_args.as_dict())

    # Setting random seed
    if training_args.manual_seed:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)

    train_dataset = dataset.TransDataset(
        training_args.input_size,
        training_args.scale_size,
        rgb_data_dir=str(training_args.rgb_train_data_dir),
        thermal_data_dir=str(training_args.thermal_train_data_dir),
        transform=transforms.Compose(
            [
                transforms.Resize((training_args.scale_size, training_args.scale_size)),
                transforms.RandomCrop(
                    (training_args.input_size, training_args.input_size), padding=1
                ),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        prior_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    val_dataset = dataset.TransDataset(
        training_args.input_size,
        training_args.scale_size,
        rgb_data_dir=str(training_args.rgb_val_data_dir),
        thermal_data_dir=str(training_args.thermal_val_data_dir),
        transform=transforms.Compose(
            [
                transforms.Resize((training_args.scale_size, training_args.scale_size)),
                transforms.RandomCrop(
                    (training_args.input_size, training_args.input_size), padding=1
                ),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        prior_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=training_args.num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=training_args.eval_batch_size,
        shuffle=False,
        num_workers=training_args.num_workers,
    )

    translator = Translator(training_args, client)
    # train
    translator.train_parameters(train_loader=train_loader, val_loader=val_loader)
    # log val_images
    translator.log_images(val_loader)

    model_filename = "translator-last"
    translator.save_checkpoint(training_args.niter_total, model_filename)

    # logging model
    train_loader_iter = iter(train_loader)
    data = next(train_loader_iter)
    x = data["RGB"].to(translator._device)
    signature = infer_signature(
        x.cpu().numpy(), translator._net(x).detach().cpu().numpy()
    )
    mlflow.pytorch.log_model(
        translator._net,
        model_filename,
        signature=signature,
        registered_model_name=training_args.dataset_name,
    )

    mlflow.end_run()


if __name__ == "__main__":
    train()
