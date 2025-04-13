from torchvision.datasets import HMDB51, UCF101
from pytorchvideo.data.hmdb51 import Hmdb51
from pytorchvideo.data.ucf101 import Ucf101
from pytorchvideo.data.clip_sampling import UniformClipSampler
import os
import pickle
from constants import UCF101_CLASSES


class CustomUCF101(UCF101):
    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label


class CustomHMDB51(HMDB51):
    def __getitem__(self, idx):
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        return video, class_index


def load_lazy(dataset):
    with open(f"cache/dataset_train_{dataset}.pickle", "rb") as file:
        dataset_train = pickle.load(file)
    with open(f"cache/dataset_valid_{dataset}.pickle", "rb") as file:
        dataset_valid = pickle.load(file)
    return dataset_train, dataset_valid


def get_dataset(dataset: str, ds_transform_train, ds_transform_valid, test_transform, lazy=True):
    if dataset not in ["hmdb51", "ucf101"]:
        raise ValueError(f"Wrong dataset {dataset}")
    if dataset == "hmdb51":
        dataset_test = Hmdb51(
            data_path=r"datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
            clip_sampler=UniformClipSampler(1000),  # sample whole clips
            decode_audio=False,
            video_path_prefix="datasets/hmdb/data",
            split_type="test",
            transform=test_transform,
        )
        if lazy and all(
            [
                os.path.exists(f"cache/dataset_train_{dataset}.pickle"),
                os.path.exists(f"cache/dataset_valid_{dataset}.pickle"),
            ]
        ):
            dataset_train, dataset_valid = load_lazy(dataset)
        else:
            dataset_train = CustomHMDB51(
                root="datasets/hmdb/data",
                annotation_path="datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
                frames_per_clip=64,
                step_between_clips=64,
                num_workers=10,
                transform=ds_transform_train,
                train=True,
                output_format="TCHW",
            )
            dataset_valid = CustomHMDB51(
                root="datasets/hmdb/data",
                annotation_path="datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
                frames_per_clip=64,
                step_between_clips=64,
                num_workers=10,
                transform=ds_transform_valid,
                train=False,
                output_format="TCHW",
            )
    elif dataset == "ucf101":
        dataset_test = Ucf101(
            data_path=r"datasets/ucf101/test_train_splits/ucfTrainTestlist/testlist01.txt",
            clip_sampler=UniformClipSampler(1000),  # sample whole clips
            decode_audio=False,
            video_path_prefix="datasets/ucf101/data/UCF-101",
            transform=test_transform,
        )
        if lazy and all(
            [
                os.path.exists(f"cache/dataset_train_{dataset}.pickle"),
                os.path.exists(f"cache/dataset_valid_{dataset}.pickle"),
            ]
        ):
            dataset_train, dataset_valid = load_lazy(dataset)
        else:
            dataset_train = CustomUCF101(
                root="datasets/ucf101/data/UCF-101",
                annotation_path="datasets/ucf101/test_train_splits/ucfTrainTestlist",
                frames_per_clip=64,
                step_between_clips=64,
                num_workers=10,
                transform=ds_transform_train,
                train=True,
                output_format="TCHW",
            )
            dataset_valid = CustomUCF101(
                root="datasets/ucf101/data/UCF-101",
                annotation_path="datasets/ucf101/test_train_splits/ucfTrainTestlist",
                frames_per_clip=64,
                step_between_clips=64,
                num_workers=10,
                transform=ds_transform_valid,
                train=False,
                output_format="TCHW",
            )
    with open(f"cache/dataset_train_{dataset}.pickle", "wb") as file:
        pickle.dump(dataset_train, file)
    with open(f"cache/dataset_valid_{dataset}.pickle", "wb") as file:
        pickle.dump(dataset_valid, file)
    if dataset == "ucf101":
        num_labels = 101
    elif dataset == "hmdb51":
        num_labels = 51

    dataset_train.transform = ds_transform_train
    dataset_valid.transform = ds_transform_valid
    dataset_test._transform = test_transform

    return dataset_train, dataset_valid, dataset_test, num_labels
