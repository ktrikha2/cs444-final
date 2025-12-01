import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset.bdd_dataset import BDDDetectionDataset
from src.models.detector_head import make_model
from src.training.coco_eval import COCODetectionEvaluator


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_ds = BDDDetectionDataset(
        cfg['data']['images']['val'],
        cfg['data']['annotations']['val']
    )

    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = make_model(cfg)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    evaluator = COCODetectionEvaluator(cfg['data']['annotations']['val'])

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for pred, tgt in zip(outputs, targets):
                evaluator.process(int(tgt['image_id']), pred)

    evaluator.evaluate()


if __name__ == '__main__':
    main()
