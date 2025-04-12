import torch
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
from pytorchvideo.transforms import UniformTemporalSubsample, ApplyTransformToKey


class Div255(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample / 255


class SplitEqually(object):
    def __init__(self, num_frames, temp_dim=0):
        self.num_frames = num_frames
        self.temp_dim = temp_dim

    def __call__(self, sample):
        temp_steps = sample.shape[0]
        seqs = list(torch.split(sample, dim=self.temp_dim, split_size_or_sections=self.num_frames))
        if seqs[-1].shape[0] != self.num_frames:
            seqs[-1] = sample[temp_steps - self.num_frames :]

        return torch.stack(seqs)


def get_transforms(config, CLASSES_LIST, dataset_name):
    if not config["augmentations"]:
        ds_transform_train = Compose(
            transforms=[
                UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=0),
                RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
                Div255(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    if config["augmentations"] and not config["advanced_augmentations"]:
        ds_transform_train = Compose(
            transforms=[
                UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=0),
                RandomHorizontalFlip(p=0.5),
                RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
                Div255(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    if config["augmentations"] and config["advanced_augmentations"]:
        ds_transform_train = Compose(
            transforms=[
                UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=0),
                RandomHorizontalFlip(p=0.5),
                ScaleJitter(target_size=(240, 320), scale_range=(0.85, 1.33)),
                RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
                RandomApply(p=0.8, transforms=[ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)]),
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

    def convert_dtype(x):
        return x.type(torch.uint8)

    def swap_dims(x):
        return x.permute(1, 0, 2, 3)

    def get_label_idx(x, CLASSES_LIST=CLASSES_LIST):
        return CLASSES_LIST.index(x)

    test_transforms_list = [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    Lambda(convert_dtype),
                    Lambda(swap_dims),
                    SplitEqually(num_frames=64),
                    UniformTemporalSubsample(num_samples=8, temporal_dim=-4),
                    CenterCrop(size=(224, 224)),
                    Div255(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
    ]
    if dataset_name.strip().lower() == "hmdb51":
        test_transforms_list.append(ApplyTransformToKey(key="label", transform=Compose([get_label_idx])))
    test_transform = Compose(test_transforms_list)
    return ds_transform_train, ds_transform_valid, test_transform
