import wandb
from datasets import get_dataset
import torch
import random
import numpy as np
from utils import load_run, show_img_chw, save_run
from funcs import train_epoch, val_epoch, test
from models.vivit_custom.utils.scheduler import WarmupCosineSchedule
from transforms import get_transforms
from models.switch.utils import set_experts_param

from constants import get_classes
from vit_configs import get_vit_config
from loaders import get_loaders
from transformers import ViTForImageClassification
from pytorchvideo.transforms import MixUp
from models.models_functions import get_model, init_model, model_to_switch
from experts_initializers import init_model_routers


CONFIG = {
    "base_learning_rate": 5,
    "label_smoothing": 0.2,
    "epochs": 10,
    "frames": 8,
    "row_patches": 14,
    "col_patches": 14,
    "val_epochs": 1,
    "batch_size": 25,
    "weight_decay": 0,
    "load_run": False,
    "init_vit": True,
    "vit_base_size": "tiny",
    "cosine_schedule": False,
    "augmentations": True,
    "advanced_augmentations": True,
    "mixup": True,
    "mixup_alpha": 0.2,
    "dataset": "ucf101",
    "subsample": None,
    "switch": True,
    "switch_layers": [0, 1],
    "n_experts": 3,
    "switch_top_k": 1,
    "load_balance_alpha": 10,
    "model": "timesformer",
    "clip_duration": 2,
    "cut_layers": 6,  # how many last layers to remove or False/None
    "test_run": True,
    "pretrain_experts_epochs": 1,
    "router": "spatial_vertical",
    "freeze_net_iters": 30,
    "lock_router": False,
}


def main_func(config):
    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.define_metric("epoch_nr")
    wandb.define_metric("train_accuracy_step_top1")
    wandb.define_metric("train_accuracy_step_top5")
    wandb.define_metric("train_accuracy_epoch_top1", step_metric="epoch_nr")
    wandb.define_metric("train_accuracy_epoch_top5", step_metric="epoch_nr")
    wandb.define_metric("valid_accuracy_epoch_top1", step_metric="epoch_nr")
    wandb.define_metric("valid_accuracy_epoch_top5", step_metric="epoch_nr")
    wandb.define_metric("test_accuracy_epoch_top1", step_metric="epoch_nr")
    wandb.define_metric("test_accuracy_epoch_top5", step_metric="epoch_nr")
    wandb.define_metric("valid_loss_epoch", step_metric="epoch_nr")
    wandb.define_metric("train_loss_step")
    wandb.define_metric("train_loss_epoch", step_metric="epoch_nr")
    wandb.define_metric("train_load_balancing_loss_step")
    wandb.define_metric("train_classification_loss_step")
    wandb.define_metric("train_load_balancing_loss_epoch", step_metric="epoch_nr")
    wandb.define_metric("train_classification_loss_epoch", step_metric="epoch_nr")

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

    switch_top_k = config["switch_top_k"]
    if config["pretrain_experts_epochs"] and config["switch"]:
        config["switch_top_k"] = 1
    model, model_config = get_model(config["model"], num_labels, VIT_CONFIG, config)

    if config["switch"]:
        model = model_to_switch(config["model"], model, config, model_config)

    if config["pretrain_experts_epochs"] and config["switch"]:
        model = init_model_routers(model, config, dataset_train)

    model.to(device)

    if config["init_vit"]:
        model = init_model(config["model"], model, vit, model_config.num_hidden_layers, config)

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

    if config["cosine_schedule"]:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(0.06 * len(training_loader) * epochs),
            t_total=len(training_loader) * epochs,
        )
    else:
        scheduler = None
    loss = torch.nn.functional.cross_entropy

    if config["mixup"]:
        mixup = MixUp(
            alpha=config["mixup_alpha"], num_classes=len(CLASSES_LIST), label_smoothing=config["label_smoothing"]
        )
    else:
        mixup = None

    if config["load_run"]:
        model, optimizer, prev_epoch = load_run(model, optimizer, config["load_run"])
    else:
        prev_epoch = 0

    max_val_acc = 0
    max_val_epoch = 0
    for epoch in range(1 + prev_epoch, 1 + epochs):

        train_epoch(
            model,
            device,
            training_loader,
            epoch,
            optimizer,
            loss,
            label_smoothing,
            mixup,
            scheduler,
            config,
            wandb,
        )

        if epoch % config["val_epochs"] == 0 and epoch > config.get("pretrain_experts_epochs", 0):
            acc = val_epoch(model, validation_loader, device, loss, config, wandb)
            if acc > max_val_acc:
                max_val_acc = acc
                max_val_epoch = epoch
                save_run(model, optimizer, wandb.run.name, epoch)

        if config["pretrain_experts_epochs"] and epoch == config["pretrain_experts_epochs"] and config["switch"]:
            model = set_experts_param(model, config["switch_layers"], "top_k", switch_top_k)
            model = set_experts_param(model, config["switch_layers"], "route", False)

    print(f"Running test for model {wandb.run.name}, Epoch: {max_val_epoch}, Max acc: {max_val_acc}")
    model, _, _ = load_run(model, optimizer, wandb.run.name, max_val_epoch)
    test(model, test_loader, device, config, wandb)


if __name__ == "__main__":
    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")
    wandb.init(project="msc", config=CONFIG)

    main_func(CONFIG)
