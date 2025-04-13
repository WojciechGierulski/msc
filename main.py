import wandb
from datasets import get_dataset
import torch
import random
import numpy as np
from utils import load_run, show_img_chw
from funcs import train_epoch, val_epoch, test_epoch
from weight_inits import init_timesformer
from models.vivit_custom.utils.scheduler import WarmupCosineSchedule
from transforms import get_transforms

from constants import get_classes
from vit_configs import get_vit_config
from loaders import get_loaders
from transformers import TimesformerConfig, TimesformerForVideoClassification
from transformers import ViTForImageClassification
from pytorchvideo.transforms import MixUp


if __name__ == "__main__":

    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")

    config = {
        "base_learning_rate": 0.0001,
        "label_smoothing": 0.0,
        "epochs": 10,
        "frames": 8,
        "val_epochs": 1,
        "batch_size": 50,
        "weight_decay": 0,
        "load_run": False,
        "init_vit": True,
        "vit_base_size": "tiny",
        "cosine_schedule": False,
        "augmentations": True,
        "advanced_augmentations": True,
        "mixup": False,
        "mixup_alpha": 0.0,
        "dataset": "ucf101",
        "subsample": None,
    }

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

    CLASSES_LIST = get_classes(config["dataset"])

    VIT_CONFIG = get_vit_config(config["vit_base_size"])

    ds_transform_train, ds_transform_valid, test_transform = get_transforms(config, CLASSES_LIST, config["dataset"])

    dataset_train, dataset_valid, dataset_test, num_labels = get_dataset(
        config["dataset"],
        ds_transform_train,
        ds_transform_valid,
        test_transform,
        lazy=True,
        subsample=config["subsample"],
        num_workers=12
    )

    vit = ViTForImageClassification.from_pretrained(f"WinKawaks/vit-{config['vit_base_size']}-patch16-224")

    timesformer_config = TimesformerConfig(num_frames=config["frames"], **VIT_CONFIG, num_labels=num_labels)
    timesformer = TimesformerForVideoClassification(timesformer_config)
    timesformer.to(device)

    if config["init_vit"]:
        timesformer = init_timesformer(timesformer, vit, VIT_CONFIG["num_hidden_layers"])

    training_loader, validation_loader, test_loader = get_loaders(
        dataset_train, dataset_valid, dataset_test, SEED, config["batch_size"], n_workers=12
    )

    epochs = config["epochs"]
    label_smoothing = config["label_smoothing"]

    optimizer = torch.optim.SGD(
        timesformer.parameters(), lr=config["base_learning_rate"], momentum=0.9, weight_decay=config["weight_decay"]
    )

    if config["cosine_schedule"]:
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=int(0.06 * len(training_loader) * epochs), t_total=len(training_loader) * epochs
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
        timesformer, optimizer, prev_epoch = load_run(timesformer, optimizer, config["load_run"])
    else:
        prev_epoch = 0

    for epoch in range(1 + prev_epoch, 1 + prev_epoch + epochs):
        train_epoch(
            timesformer,
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
            val_epoch(timesformer, validation_loader, device, loss, wandb)
    test_epoch(timesformer, test_loader, device, wandb)
