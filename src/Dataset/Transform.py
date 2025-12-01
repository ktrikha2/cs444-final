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
            boxes[:, 0] = w - (boxes[:, 0] + boxes[:, 2])
            target['boxes'] = boxes
    return image, target


def filter_invalid_boxes(target):
    boxes = target["boxes"]

    # Keep only boxes with positive width & height
    keep = (boxes[:, 2] > 1) & (boxes[:, 3] > 1)

    target["boxes"]  = boxes[keep]
    target["labels"] = target["labels"][keep]

    return target


def compose_transforms():
    def transform(image, target):
        image, target = random_horizontal_flip(image, target, p=0.5)
        target = filter_invalid_boxes(target)
        return image, target
    return transform
