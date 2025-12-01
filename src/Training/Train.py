import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.Dataset.Dataset import BDDDetectionDataset
from src.Dataset.Transform import compose_transforms
from src.Models.Detector_head import make_model
from src.Training.Utils import set_seed, save_checkpoint


def collate_fn(batch):
    return tuple(zip(*batch))


def train_loop(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    return running_loss / len(data_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))

    device = torch.device(
        cfg['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ----------------------------------------------------
    # Load dataset
    # ----------------------------------------------------
    train_ds = BDDDetectionDataset(
        cfg['data']['images']['train'],
        cfg['data']['annotations']['train'],
        transforms=compose_transforms()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn
    )

    # ----------------------------------------------------
    # Build model
    # ----------------------------------------------------
    model = make_model(cfg)

    # BDD100K has 10 object classes → +1 background = 11
    num_classes = 10 + 1  

    in_features_cls = model.roi_heads.box_predictor.cls_score.in_features
    in_features_bbox = model.roi_heads.box_predictor.bbox_pred.in_features

    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features_cls, num_classes
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features_bbox, num_classes * 4
    )

    model.to(device)

    # ----------------------------------------------------
    # Optimizer
    # ----------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['training']['lr'], weight_decay=1e-4
    )

    # ----------------------------------------------------
    # Training loop
    # ----------------------------------------------------
    num_epochs = cfg['training']['epochs']

    for epoch in range(num_epochs):
        loss = train_loop(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")

        save_path = os.path.join(cfg['training']['checkpoint_dir'], f"model_epoch{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, save_path)


if __name__ == "__main__":
    main()