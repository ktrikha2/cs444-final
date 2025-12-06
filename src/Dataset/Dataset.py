import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class BDDDetectionDataset(Dataset):
    """A minimal dataset that expects COCO-style bounding-box JSONs (x,y,w,h)
    and an image folder. If your annotations are in BDD format, convert them
    to COCO-like detection JSONs first.
    """

    def __init__(self, images_dir, ann_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Expect COCO-style structure: images:[], annotations:[]
        self.images = {img['id']: img for img in data['images']}
        self.imgs_by_idx = list(self.images.keys())

        # group annotations by image_id
        self.anns = {}
        for ann in data['annotations']:
            self.anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.imgs_by_idx)

    def __getitem__(self, idx):
        img_id = self.imgs_by_idx[idx]
        meta = self.images[img_id]
        path = os.path.join(self.images_dir, meta['file_name'])
        img = Image.open(path).convert('RGB')

        # build target
        annos = self.anns.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in annos:
            # COCO bbox -> x, y, w, h
            x, y, w, h = a['bbox']
            boxes.append([x, y, w, h])
            labels.append(a['category_id'] - 1)
            areas.append(a.get('area', w * h))
            iscrowd.append(a.get('iscrowd', 0))

        to_tensor = T.ToTensor()
        image = to_tensor(img)

        target = {}
        if boxes:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['image_id'] = torch.tensor([img_id])
            target['area'] = torch.as_tensor(areas, dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # empty tensors for images with no objects
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['image_id'] = torch.tensor([img_id])
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
