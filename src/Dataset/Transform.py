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

def xywh_to_xyxy(boxes):
    # boxes: [N, 4] = [x, y, w, h]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)

def normalize_image(image):
    """Apply ImageNet normalization for pretrained Swin"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image - mean) / std

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
        #target["boxes"] = xywh_to_xyxy(target["boxes"])
        target = filter_invalid_boxes(target)
        image = normalize_image(image)
        return image, target
    return transform