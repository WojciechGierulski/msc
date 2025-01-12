import wandb
from datasets import get_dataset
import torch
from utils import save_run, load_run
from funcs import train_epoch, val_epoch, test_epoch
from weight_inits import init_timesformer
from torch.utils.data import DataLoader
from models.vivit_custom.utils.scheduler import WarmupCosineSchedule
from transforms import get_transforms

from constants import get_classes
from transformers import TimesformerConfig, TimesformerForVideoClassification
from transformers import ViTForImageClassification
from pytorchvideo.transforms import MixUp


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")

    config = {
        "base_learning_rate": 0.0001,
        "label_smoothing": 0.1,
        "epochs": 10,
        "layers": 12,
        "frames": 8,
        "val_epochs": 1,
        "batch_size": 8,
        "load_run": False,
        "dataset": "ucf101",
        "init_vit": True,
        "augmentations": True,
        "advanced_augmentations": True,
        "mixup": True,
        "mixup_alpha": True,
        "cosine_schedule": True,
        "weight_decay": 1e-5
    }

    wandb.init(project="msc", config=config)

    wandb.define_metric("epoch_nr")
    wandb.define_metric("train_accuracy_step_top1")
    wandb.define_metric("train_accuracy_step_top5")
    wandb.define_metric("train_accuracy_epoch_top1", step_metric="epoch_nr")
    wandb.define_metric("train_accuracy_epoch_top5", step_metric="epoch_nr")
    wandb.define_metric("valid_accuracy_epoch_top1", step_metric="epoch_nr")
    wandb.define_metric("valid_accuracy_epoch_top5", step_metric="epoch_nr")
    wandb.define_metric("valid_loss_epoch", step_metric="epoch_nr")
    wandb.define_metric("train_loss_step")
    wandb.define_metric("train_loss_epoch", step_metric="epoch_nr")


    CLASSES_LIST = get_classes(config['dataset'])

    ds_transform_train, ds_transform_valid, test_transform = get_transforms(config, CLASSES_LIST)

    dataset_train, dataset_valid, dataset_test = get_dataset(
        config["dataset"], ds_transform_train, ds_transform_valid, test_transform
    )

    vit = ViTForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224")

    timesformer_config = TimesformerConfig(
        num_frames=config["frames"],
        hidden_size=384,
        num_attention_heads=6,
        intermediate_size=384 * 4,
        num_labels=51,
        num_hidden_layers=config["layers"],
    )
    timesformer = TimesformerForVideoClassification(timesformer_config)
    timesformer.to(device)
    
    if config['init_vit']:
        timesformer = init_timesformer(timesformer, vit, config["layers"])

    training_loader = DataLoader(dataset=dataset_train, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(dataset=dataset_valid, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=1)

    epochs = config["epochs"]
    label_smoothing = config["label_smoothing"]

    optimizer = torch.optim.SGD(
        timesformer.parameters(), lr=config["base_learning_rate"], momentum=0.9, weight_decay=config['weight_decay']
    )

    if config['cosine_schedule']:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(0.06 * len(training_loader)), t_total=len(training_loader) * epochs)
    else:
        scheduler = None
    loss = torch.nn.functional.cross_entropy

    if config['mixup']:
        mixup = MixUp(alpha=config['mixup_alpha'], label_smoothing=config['label_smoothing'], num_classes=len(CLASSES_LIST))
    else:
        mixup = None

    if config["load_run"]:
        timesformer, optimizer, prev_epoch = load_run(timesformer, optimizer, config["load_run"])
    else:
        prev_epoch = 0

    for epoch in range(1 + prev_epoch, 1 + prev_epoch + epochs):
        train_epoch(timesformer, device, training_loader, epoch, optimizer, loss, label_smoothing, mixup, scheduler, wandb)

        if epoch % config["val_epochs"] == 0:
            val_epoch(timesformer, validation_loader, device, loss, wandb)
        break

    test_epoch(timesformer, test_loader, device, wandb)
