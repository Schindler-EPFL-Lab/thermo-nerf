import random
import sys
from dataclasses import dataclass

import mlflow
import torch
import torchvision.transforms as transforms
import tyro
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path
sys.path.append(".")
sys.path.append("./nerfstudio")
from rebel_nerf.image_translator.config.parameters import (  # noqa: E402
    TranslatorParameters,
)
from rebel_nerf.image_translator.data import dataset  # noqa: E402
from rebel_nerf.image_translator.model.translator import Translator  # noqa: E402


@dataclass
class TestParam(TranslatorParameters):
    model_uri: str


def test():
    mlflow.start_run()
    client = MlflowClient()
    test_args = tyro.cli(TestParam)

    # Setting random seed
    if test_args.manual_seed:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)

    test_dataset = dataset.TransDataset(
        test_args.input_size,
        test_args.scale_size,
        rgb_data_dir=str(test_args.rgb_test_data_dir),
        thermal_data_dir=str(test_args.thermal_test_data_dir),
        transform=transforms.Compose(
            [transforms.Resize((test_args.input_size, test_args.input_size))]
        ),
        prior_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_args.eval_batch_size,
        shuffle=False,
        num_workers=test_args.num_workers,
    )

    # build a translator
    translator = Translator(test_args, client)

    # load pre-trained ckpt
    model = mlflow.pytorch.load_model(test_args.model_uri)
    translator.load_checkpoint(model.state_dict())

    # run test
    translator.test(test_loader)


if __name__ == "__main__":
    test()
