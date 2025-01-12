from torchvision.datasets import HMDB51, UCF101
from pytorchvideo.data.hmdb51 import Hmdb51
from pytorchvideo.data.ucf101 import Ucf101
from pytorchvideo.data.clip_sampling import UniformClipSampler


def get_dataset(dataset: str, ds_transform_train, ds_transform_valid, test_transform):
    if dataset not in ["hmdb51", "ucf101"]:
        raise ValueError(f"Wrong dataset {dataset}")
    if dataset == 'hmdb51':
        dataset_train = HMDB51(
            root="datasets/hmdb/data",
            annotation_path="datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
            frames_per_clip=64,
            step_between_clips=64,
            num_workers=4,
            transform=ds_transform_train,
            train=True,
            output_format="TCHW",
        )
        dataset_valid = HMDB51(
            root="datasets/hmdb/data",
            annotation_path="datasets/hmdb/test_train_splits/testTrainMulti_7030_splits",
            frames_per_clip=64,
            step_between_clips=64,
            num_workers=4,
            transform=ds_transform_valid,
            train=False,
            output_format="TCHW",
        )
        dataset_test = Hmdb51(
            data_path=r"datasets\hmdb\test_train_splits\testTrainMulti_7030_splits",
            clip_sampler=UniformClipSampler(1000),  # sample whole clips
            decode_audio=False,
            video_path_prefix="datasets/hmdb/data",
            split_type="test",
            transform=test_transform,
        )
    elif dataset == "ucf101":
        dataset_train = UCF101(
            root="datasets/ucf101/data/UCF-101",
            annotation_path="datasets/ucf101/test_train_splits/ucfTrainTestlist",
            frames_per_clip=64,
            step_between_clips=64,
            num_workers=4,
            transform=ds_transform_train,
            train=True,
            output_format="TCHW",
        )
        dataset_valid = UCF101(
            root="datasets/ucf101/data/UCF-101",
            annotation_path="datasets/ucf101/test_train_splits/ucfTrainTestlist",
            frames_per_clip=64,
            step_between_clips=64,
            num_workers=4,
            transform=ds_transform_valid,
            train=False,
            output_format="TCHW",
        )
        dataset_test = Ucf101(
            data_path=r"datasets/ucf101/data/UCF-101",
            clip_sampler=UniformClipSampler(1000),  # sample whole clips
            decode_audio=False,
            video_path_prefix="datasets/ucf101/data/UCF-101",
            transform=test_transform,
        )
    return dataset_train, dataset_valid, dataset_test