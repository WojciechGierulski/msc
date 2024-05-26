from typing import Any
from torchvision.transforms.v2 import Resize
import torch


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
