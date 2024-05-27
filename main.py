import wandb
from torchvision.datasets import HMDB51
import torch
import tqdm
import torch
from utils import save_run, load_run
from funcs import train_epoch, val_epoch, test_epoch
from transforms import Div255, SplitEqually
from torchvision.transforms.v2 import (
    ScaleJitter,
    RandAugment,
    RandomCrop,
    Compose,
    ColorJitter,
    RandomHorizontalFlip,
    Normalize,
    RandomApply,
    UniformTemporalSubsample,
    CenterCrop,
    Lambda,
)
from weight_inits import init_timesformer
from torch.utils.data import DataLoader
from models.vivit_custom.utils.scheduler import WarmupCosineSchedule
from pytorchvideo.transforms import ApplyTransformToKey
from pytorchvideo.data.hmdb51 import Hmdb51
from pytorchvideo.data.clip_sampling import UniformClipSampler
from pytorchvideo.transforms import UniformTemporalSubsample, ApplyTransformToKey
from constants import HMDB_CLASSES
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


    ds_transform_train = Compose(
        transforms=[
            UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=0),
            RandomHorizontalFlip(p=0.5),
            ScaleJitter(target_size=(240, 320), scale_range=(0.9, 1.33)),
            RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
            RandomApply(p=0.8, transforms=[ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1)]),
            RandAugment(num_ops=2, magnitude=10, fill=128),
            Div255(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_transform_valid = Compose(
        transforms=[
            UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=0),
            CenterCrop((224, 224)),
            Div255(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_train = HMDB51(
        root="datasets/hmdb/data",
        annotation_path="datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
        frames_per_clip=640,
        step_between_clips=640,
        num_workers=4,
        transform=ds_transform_train,
        train=True,
        output_format="TCHW",
    )


    dataset_valid = HMDB51(
        root="datasets/hmdb/data",
        annotation_path="datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
        frames_per_clip=640,
        step_between_clips=640,
        num_workers=4,
        transform=ds_transform_valid,
        train=False,
        output_format="TCHW",
    )

    test_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Lambda(lambda x: x.type(torch.uint8)),
                        Lambda(lambda x: x.permute(1, 0, 2, 3)),
                        SplitEqually(num_frames=64),
                        UniformTemporalSubsample(num_samples=8, temporal_dim=-4),
                        CenterCrop(size=(224, 224)),
                        Div255(),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                ),
            ),
            ApplyTransformToKey(key="label", transform=Compose([lambda x: HMDB_CLASSES.index(x)])),
        ]
    )

    dataset_test = Hmdb51(
        data_path=r"datasets\hmdb\test_train_splits\testTrainMulti_7030_splits",
        clip_sampler=UniformClipSampler(1000),  # sample whole clips
        decode_audio=False,
        video_path_prefix="datasets/hmdb/data",
        split_type="test",
        transform=test_transform,
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

    timesformer = init_timesformer(timesformer, vit, config["layers"])

    training_loader = DataLoader(dataset=dataset_train, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(dataset=dataset_valid, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=1)


    epochs = config["epochs"]
    label_smoothing = config["label_smoothing"]

    optimizer = torch.optim.SGD(timesformer.parameters(), lr=config["base_learning_rate"], momentum=0.9, weight_decay=1e-5)
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(0.06 * len(training_loader)), t_total=len(training_loader) * epochs)
    loss = torch.nn.functional.cross_entropy

    mixup = MixUp(alpha=0.2, label_smoothing=0, num_classes=51)

    if config["load_run"]:
        timesformer, optimizer, prev_epoch = load_run(timesformer, optimizer, config["load_run"])
    else:
        prev_epoch = 0


    for epoch in range(1 + prev_epoch, 1 + prev_epoch + epochs):
        train_epoch(timesformer, device, training_loader, epoch, optimizer, loss, label_smoothing, mixup, wandb)

        if epoch % config["val_epochs"] == 0:
            val_epoch(timesformer, validation_loader, device, loss, wandb)
        break


    test_epoch(timesformer, test_loader, device, wandb)
