import wandb
from datasets import get_dataset
import torch
import random
import numpy as np
from utils import load_run, show_img_chw, save_run
from funcs import train_epoch, val_epoch, test
from models.vivit_custom.utils.scheduler import WarmupCosineSchedule
from transforms import get_transforms

from constants import get_classes
from vit_configs import get_vit_config
from loaders import get_loaders
from transformers import ViTForImageClassification
from pytorchvideo.transforms import MixUp
from models.models_functions import get_model, init_model, model_to_switch


CONFIG = {
    "base_learning_rate": 0.0005,
    "label_smoothing": 0.2,
    "epochs": 10,
    "frames": 8,
    "val_epochs": 1,
    "batch_size": 24,
    "weight_decay": 0,
    "load_run": False,
    "init_vit": True,
    "vit_base_size": "small",
    "cosine_schedule": False,
    "augmentations": True,
    "advanced_augmentations": True,
    "mixup": True,
    "mixup_alpha": 0.2,
    "dataset": "hmdb51",
    "subsample": None,
    "switch": False,
    "switch_layers": [1, 3, 5, 7, 9, 11],
    "n_experts": 3,
    "load_balance_alpha": 10,
    "model": "timesformer",
    "clip_duration": 2,
}


def main_func(config):
    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")
    wandb.init(project="msc", config=config)

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

    for epoch in range(1 + prev_epoch, 1 + prev_epoch + epochs):
        max_val_acc = 0
        max_val_epoch = 0
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

        if epoch % config["val_epochs"] == 0:
            acc = val_epoch(model, validation_loader, device, loss, wandb)
            if acc > max_val_acc:
                max_val_acc = acc
                max_val_epoch = epoch
                save_run(model, optimizer, wandb.run.name, epoch)

    print(f"Running test for model {wandb.run.name}, Epoch: {max_val_epoch}, Max acc: {max_val_acc}")
    model, _, _ = load_run(model, optimizer, wandb.run.name, max_val_epoch)
    test(model, test_loader, device, wandb)


if __name__ == "__main__":
    main_func(CONFIG)
