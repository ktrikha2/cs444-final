import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.Dataset.Dataset import BDDDetectionDataset
from src.Dataset.Transform import compose_transforms
from src.Models.SwinDETR import build_swin_detr
from src.Training.Utils import set_seed, save_checkpoint
from src.Losses.HungarianMatcher import HungarianMatcher
from src.Losses.Critertion import SetCriterion


def collate_fn(batch):
    # returns (list_of_images, list_of_targets)
    return tuple(zip(*batch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def train_epoch(model, criterion, data_loader, optimizer, device, weight_dict):
    model.train()
    criterion.train()
    running_loss = 0.0



    #TIMING TO SEE BOTTLENECK
    epoch_t0 = time.time()

    data_time = 0.0
    forward_time = 0.0
    loss_time = 0.0
    backward_time = 0.0
    optim_time = 0.0

    for images, targets in data_loader:
        # images: tuple of tensors [3,H,W]; targets: tuple of dicts
        t_data_start = time.time()


        images = [img.to(device) for img in images]
        batch_tensor = torch.stack(images, dim=0)  # [B,3,H,W]

        processed_targets = []
        for img, t in zip(images, targets):
            h, w = img.shape[-2:]
            tgt = {k: v.to(device) for k, v in t.items()}
            tgt["img_size"] = torch.tensor([h, w], dtype=torch.float32, device=device)
            processed_targets.append(tgt)
        data_time += time.time() - t_data_start #data loading time

        t_fwd = time.time()
        outputs = model(batch_tensor)
        forward_time += time.time() - t_fwd
       
        t_loss = time.time()
        loss_dict = criterion(outputs, processed_targets)
        # weighted sum of all losses
        loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys())
        loss_time += time.time() - t_loss
        
        t_bwd = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time += time.time() - t_bwd
        
        t_opt = time.time()
        optimizer.step()
        optim_time += time.time() - t_opt
        running_loss += loss.item()
    epoch_total = time.time() - epoch_t0
    print(
        f"Epoch Timing | "
        f"total={epoch_total:.1f}s | "
        f"data={data_time:.1f}s | "
        f"fwd={forward_time:.1f}s | "
        f"loss={loss_time:.1f}s | "
        f"bwd={backward_time:.1f}s | "
        f"opt={optim_time:.1f}s",
        flush=True
    )
    return running_loss / len(data_loader)


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = torch.device(
        cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    train_ds = BDDDetectionDataset(
        cfg["data"]["images"]["train"],
        cfg["data"]["annotations"]["train"],
        transforms=compose_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = build_swin_detr(cfg)
    model.to(device)

    num_classes = cfg["model"]["num_classes"]  # BDD: 10

    # -----------------------------
    # Loss / Matcher
    # -----------------------------
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
    ).to(device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=1e-4
    )

    num_epochs = cfg["training"]["epochs"]

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(num_epochs):
        loss = train_epoch(model, criterion, train_loader, optimizer, device, weight_dict)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")

        if (epoch + 1) % cfg["training"]["checkpoint_freq"] == 0:
            save_path = os.path.join(
                cfg["training"]["checkpoint_dir"], f"swin_detr_epoch{epoch+1}.pth"
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )


if __name__ == "__main__":
    main()
