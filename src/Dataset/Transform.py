import torchvision.transforms.functional as F
import random
import torch


# Keep transforms minimal. The dataset above already returns tensors.

def random_horizontal_flip(image, target, p=0.5):
    if random.random() < p:
        image = F.hflip(image)
        if 'boxes' in target and target['boxes'].numel() > 0:
            w = image.shape[2]
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes
    return image, target


def compose_transforms():
    def transform(image, target):
        image, target = random_horizontal_flip(image, target, p=0.5)
        return image, target
    return transform
