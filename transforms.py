from typing import Any
from torchvision.transforms.v2 import Resize

class ToVideo(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class ResizeVideo(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.transform = Resize(output_size)

    def __call__(self, sample):
        result = self.transform(sample)
        return result


class Div255(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample/255
