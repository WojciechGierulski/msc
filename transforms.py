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
from pytorchvideo.transforms import UniformTemporalSubsample, ApplyTransformToKey, MixUp


class Div255(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample / 255


class SplitTo3(object):
    def __init__(self, clip_frames):
        self.clip_frames = clip_frames
        pass

    def __call__(self, a):
        clip_frames = self.clip_frames
        frames = a.shape[0]

        if frames < clip_frames:
            repeated = a[-1].unsqueeze(0).repeat((clip_frames - frames, 1, 1, 1))
            a = torch.cat((a, repeated), dim=0)
            a = a.unsqueeze(0)

        frames = a.shape[0]
        mid_frame = frames // 2
        left_idx_1 = max(0, mid_frame - clip_frames // 2 - clip_frames)
        right_idx_1 = left_idx_1 + clip_frames
        left_idx_2 = mid_frame - clip_frames // 2
        right_idx_2 = mid_frame + clip_frames // 2
        right_idx_3 = min(mid_frame + clip_frames // 2 + clip_frames, a.shape[0])
        left_idx_3 = right_idx_3 - clip_frames
        a = torch.stack((a[left_idx_1:right_idx_1], a[left_idx_2:right_idx_2], a[left_idx_3:right_idx_3]))

        return a


def get_transforms(config, CLASSES_LIST, dataset_name):

    def convert_dtype(x):
        return x.type(torch.uint8)

    def swap_dims(x):
        return x.permute(1, 0, 2, 3)

    def get_label_idx(x, CLASSES_LIST=CLASSES_LIST):
        return CLASSES_LIST.index(x)

    def subtract_one(x):
        return x - 1

    if not config["augmentations"]:
        train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=1),
                    Lambda(swap_dims),
                    Lambda(convert_dtype),
                    RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
                    Div255(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
    if config["augmentations"] and not config["advanced_augmentations"]:
        train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=1),
                    Lambda(swap_dims),
                    Lambda(convert_dtype),
                    RandomHorizontalFlip(p=0.5),
                    RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
                    Div255(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
    if config["augmentations"] and config["advanced_augmentations"]:
        train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=1),
                    Lambda(swap_dims),
                    Lambda(convert_dtype),
                    RandomHorizontalFlip(p=0.5),
                    ScaleJitter(target_size=(256, 256), scale_range=(0.95, 1.33)),
                    RandomCrop(size=(224, 224), pad_if_needed=True, fill=128),
                    RandomApply(
                        p=0.8, transforms=[ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)]
                    ),
                    RandAugment(num_ops=2, magnitude=10, fill=128),
                    Div255(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )

    valid_transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=1),
                Lambda(swap_dims),
                Lambda(convert_dtype),
                CenterCrop((224, 224)),
                Div255(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    test_transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                Lambda(swap_dims),
                Lambda(convert_dtype),
                SplitTo3(64),
                UniformTemporalSubsample(num_samples=config["frames"], temporal_dim=1),
                CenterCrop((224, 224)),
                Div255(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    if dataset_name.strip().lower() == "hmdb51":
        train_transform = Compose(
            [train_transform, ApplyTransformToKey(key="label", transform=Compose([get_label_idx]))]
        )
        valid_transform = Compose(
            [valid_transform, ApplyTransformToKey(key="label", transform=Compose([get_label_idx]))]
        )
        test_transform = Compose(
            [test_transform, ApplyTransformToKey(key="label", transform=Compose([get_label_idx]))]
        )
    if dataset_name.strip().lower() == "ucf101":
        train_transform = Compose(
            [
                ApplyTransformToKey(key="label", transform=Compose([subtract_one])),
                train_transform,
            ]
        )
        valid_transform = Compose(
            [ApplyTransformToKey(key="label", transform=Compose([subtract_one])), valid_transform]
        )
        test_transform = Compose([test_transform, ApplyTransformToKey(key="label", transform=Compose([subtract_one]))])

    return train_transform, valid_transform, test_transform
