import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
import tyro
from torch.utils.data import DataLoader

from rebel_nerf.image_translator.config.parameters import TranslatorParameters
from rebel_nerf.image_translator.data import dataset
from rebel_nerf.image_translator.model.translator import Translator


def test():
    test_args = tyro.cli(TranslatorParameters)

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
            [
                transforms.Resize((test_args.scale_size, test_args.scale_size)),
                transforms.RandomCrop(
                    (test_args.input_size, test_args.input_size), padding=1
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

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_args.batch_size,
        shuffle=True,
        num_workers=test_args.num_workers,
    )

    model = Translator(test_args)

    checkpoint_path = Path(test_args.checkpoint_dir, test_args.test_ckpt_path)
    model.load_checkpoint(checkpoint_path, keep_kw_module=False)

    model.test(test_loader)


if __name__ == "__main__":
    test()
