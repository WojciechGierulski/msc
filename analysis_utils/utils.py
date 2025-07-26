import wandb
from datasets import get_dataset
import torch
import random
import numpy as np
from utils import load_run, show_img_chw, get_image_chw
from funcs import train_epoch, val_epoch, test
from transforms import get_transforms

from models.switch.utils import set_experts_param
from constants import get_classes
from vit_configs import get_vit_config
from loaders import get_loaders
from transformers import ViTForImageClassification
from pytorchvideo.transforms import MixUp
from models.models_functions import get_model, init_model, model_to_switch
import matplotlib.pyplot as plt
import os


def get_things_wandb(RUN_ID):
    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")
    wandb.init(project="msc", resume=True, id=RUN_ID)
    config = wandb.config

    if "switch_top_k" not in config:
        config["switch_top_k"] = 1
    run_model_name = wandb.run.name
    return config, run_model_name


def get_things(config, batch_size, SEED, EPOCH_NR, run_model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["batch_size"] = batch_size

    CLASSES_LIST = get_classes(config["dataset"])

    VIT_CONFIG = get_vit_config(config["vit_base_size"])

    ds_transform_train, ds_transform_valid, ds_transform_test = get_transforms(config, CLASSES_LIST, config["dataset"])

    dataset_train, dataset_valid, dataset_test, num_labels = get_dataset(
        config["dataset"],
        ds_transform_train,
        ds_transform_valid,
        ds_transform_test,
        seconds=config["clip_duration"],
        subsample=config["subsample"],
    )

    vit = ViTForImageClassification.from_pretrained(f"WinKawaks/vit-{config['vit_base_size']}-patch16-224")

    model, model_config = get_model(config["model"], num_labels, VIT_CONFIG, config)

    if config["switch"]:
        model = model_to_switch(config["model"], model, config, model_config)

    model.to(device)

    if config["init_vit"]:
        model = init_model(config["model"], model, vit, VIT_CONFIG["num_hidden_layers"], config)

    training_loader, validation_loader, test_loader = get_loaders(
        dataset_train, dataset_valid, dataset_test, SEED, config["batch_size"], n_workers=1
    )

    epochs = config["epochs"]
    label_smoothing = config["label_smoothing"]

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["base_learning_rate"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
    )

    loss = torch.nn.functional.cross_entropy

    model, optimizer, prev_epoch = load_run(model, optimizer, run_model_name, EPOCH_NR)

    model = set_experts_param(model, config["switch_layers"], "save_token_routes", True)

    return model, training_loader, validation_loader, test_loader, config


def run_for_n_iters(n_iters, model, config, validation_loader, device):
    iterr = 0

    iter_routes = {i: {"labels": [], "routes": []} for i in config["switch_layers"]}

    with torch.no_grad():
        for data in validation_loader:
            iterr += 1
            if iterr >= n_iters:
                break
            images = data["video"]
            labels = data["label"]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits

            for layer_nr in config["switch_layers"]:
                routes = model.timesformer.encoder.layer[layer_nr].intermediate.token_routes.cpu().numpy()
                iter_routes[layer_nr]["routes"].append(list(routes))
                iter_routes[layer_nr]["labels"].append(list(labels.cpu().numpy()))
    return iter_routes


def get_model_epochs(run_name):
    epochs = []
    for el in os.listdir(f"./checkpoints/{run_name}"):
        epoch_nr = el.split(".pt")[0].split("_epoch_")[1]
        epoch_nr = int(epoch_nr)
        epochs.append(epoch_nr)
    return sorted(epochs)
