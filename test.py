import wandb
from datasets import get_dataset
import torch
import random
import numpy as np
from utils import load_run, show_img_chw, save_run
from funcs import train_epoch, val_epoch, test
from models.vivit_custom.utils.scheduler import WarmupCosineSchedule
from transforms import get_transforms

from models.switch.utils import set_activate_n_expert
from constants import get_classes
from vit_configs import get_vit_config
from loaders import get_loaders
from transformers import ViTForImageClassification
from pytorchvideo.transforms import MixUp
from models.models_functions import get_model, init_model, model_to_switch

RUN_ID = ""
EPOCH_NR = 1
ACTIVATE_ONLY_N_EXPERT = 0


def main_func_resume(run_id, epoch_nr_to_load):
    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.login(key="25a61b596081a070a75d3cfc16bfad96632650fc")
    wandb.init(project="msc", resume=True, id=run_id)
    config = wandb.config
    if "switch_top_k" not in config:
        config["switch_top_k"] = 1
    run_model_name = wandb.run.name

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

    model, optimizer, prev_epoch = load_run(model, optimizer, run_model_name, epoch_nr_to_load)

    if ACTIVATE_ONLY_N_EXPERT and config["switch"]:
        model = set_activate_n_expert(model, ACTIVATE_ONLY_N_EXPERT, config["switch_layers"])

    test(model, test_loader, device, wandb)


if __name__ == "__main__":
    main_func_resume(RUN_ID, EPOCH_NR)
