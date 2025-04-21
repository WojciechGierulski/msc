import wandb
from datasets import get_dataset
import torch
import random
import numpy as np
from utils import load_run, show_img_chw
from funcs import train_epoch, val_epoch, test
from transforms import get_transforms

from constants import get_classes
from vit_configs import get_vit_config
from loaders import get_loaders
from models.models_functions import get_model, init_model, model_to_switch


LOAD_PARAMS = {"run_id": "j6ycma93", "load_epoch": 1}


def main_func(LOAD_PARAMS):
    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")
    wandb.init(id=LOAD_PARAMS["run_id"], project="msc", resume=True)
    config = wandb.config

    VIT_CONFIG = get_vit_config(config["vit_base_size"])

    wandb.define_metric("test_accuracy_epoch_top1", step_metric="epoch_nr")
    wandb.define_metric("test_accuracy_epoch_top5", step_metric="epoch_nr")

    CLASSES_LIST = get_classes(config["dataset"])

    ds_transform_train, ds_transform_valid, ds_transform_test = get_transforms(config, CLASSES_LIST, config["dataset"])

    dataset_train, dataset_valid, dataset_test, num_labels = get_dataset(
        config["dataset"],
        ds_transform_train,
        ds_transform_valid,
        ds_transform_test,
        seconds=config["clip_duration"],
        subsample=config["subsample"],
    )

    model, model_config = get_model(config["model"], num_labels, VIT_CONFIG, config)

    if config["switch"]:
        model = model_to_switch(config["model"], model, config, model_config)

    model.to(device)

    training_loader, validation_loader, test_loader = get_loaders(
        dataset_train, dataset_valid, dataset_test, SEED, config["batch_size"], n_workers=1
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["base_learning_rate"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )

    model, optimizer, prev_epoch = load_run(model, optimizer, run_name, LOAD_PARAMS["load_epoch"])

    test(model, test_loader, device, wandb)


if __name__ == "__main__":
    main_func(LOAD_PARAMS)
