import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import time
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR, MultiStepLR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.Dataset.Dataset import BDDDetectionDataset
from src.Dataset.Transform import compose_transforms
from src.Models.CNN_Swin_Detr import build_swin_detr 
from src.Training.Utils import set_seed, save_checkpoint
from src.Losses.HungarianMatcher import HungarianMatcher
from src.Losses.Critertion import SetCriterion

def collate_fn(batch):
    return tuple(zip(*batch))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def train_epoch(model, criterion, data_loader, optimizer, device, weight_dict, epoch, scaler):
    model.train()
    criterion.train()
    running_loss = 0.0

    epoch_t0 = time.time()
    data_time = forward_time = loss_time = backward_time = optim_time = 0.0
    use_cuda_amp = (device.type == "cuda")


    for batch_idx, (images, targets) in enumerate(data_loader, start=1):
        t_data_start = time.time()
        #images = [img.to(device) for img in images]
        #batch_tensor = torch.stack(images, dim=0)
        images = torch.stack(images).to(device, non_blocking=True)
        #print("Batch images mean/std:", images.mean().item(), images.std().item())
        #print("image min/max:", images.min().item(), images.max().item())
        # Process targets
        processed_targets = []
        for img, t in zip(images, targets):
            h, w = img.shape[-2:]
            tgt = {k: v.to(device) for k, v in t.items()}
            tgt["img_size"] = torch.tensor([h, w], dtype=torch.float32, device=device)
            processed_targets.append(tgt)
        data_time += time.time() - t_data_start

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        t_fwd = time.time()
        with torch.cuda.amp.autocast(enabled=use_cuda_amp):
            outputs = model(images)
        forward_time += time.time() - t_fwd
        
        # IMPORTANT: convert outputs to FP32 for Hungarian matcher
        outputs_fp32 = {
            k: v.float() if isinstance(v, torch.Tensor) else v
            for k, v in outputs.items()
        }

        t_loss = time.time()
        loss_dict = criterion(outputs_fp32, processed_targets)
        loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict)
        loss_time += time.time() - t_loss

        # Backward
        t_bwd = time.time()
        #optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        #loss.backward()
        backward_time += time.time() - t_bwd

        # Optimizer step
        t_opt = time.time()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        scaler.step(optimizer)
        scaler.update()
        #optimizer.step()
        optim_time += time.time() - t_opt

        running_loss += loss.item()

        if batch_idx % 10 == 0 or batch_idx == len(data_loader):
            avg_loss = running_loss / batch_idx
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] | Avg Loss: {avg_loss:.4f}")

    epoch_total = time.time() - epoch_t0
    print(
        f"Epoch [{epoch}] Timing | "
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
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

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
    model = build_swin_detr(cfg)  # Should return your SwinDETR backbone+neck+decoder+head
    print(model) 
    model.to(device)
    start_epoch = 1
    ckpt_path = "/work/nvme/bfdu/ktrikha/checkpoints/swindetr_LONG_CPT/swin_detr_epoch112.pth"
    
    if os.path.exists(ckpt_path):
        print(f"\n>>> Loading checkpoint from {ckpt_path}\n")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        
        # If the checkpoint has 'epoch', use it. Otherwise assume 100.
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 101 # Fallback if epoch key missing
        print(f">>> Resuming from Epoch {start_epoch}")
    else:
        print(f"\n>>> No checkpoint found at {ckpt_path}, training from scratch...\n")         
    
    
    num_classes = cfg["model"]["num_classes"]

    # -----------------------------
    # Loss / Matcher
    # -----------------------------
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=3.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 4.0}
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.25,
    ).to(device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    def is_backbone(n): return "backbone" in n
    def is_head(n): return "head" in n or "bbox_embed" in n or "class_embed" in n

    param_dicts = [
        {   # BACKBONE (Keep very low)
            "params": [p for n, p in model.named_parameters() if is_backbone(n) and p.requires_grad],
            "lr": 1e-6, 
            "weight_decay": 1e-4, 
        },
        {   # HEADS (Lower this to prevent exploding gradients/box collapse)
            "params": [p for n, p in model.named_parameters() if is_head(n) and p.requires_grad],
            "lr": 1e-4,
            "weight_decay": 1e-4, 
        },
        {   # TRANSFORMER / NECK (The rest)
            "params": [p for n, p in model.named_parameters() 
                       if not is_backbone(n) and not is_head(n) and p.requires_grad],
            "lr": 1e-4,
            "weight_decay": 1e-4, 
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts)
    scaler = GradScaler() #trying to add this for speed up 

    num_epochs = cfg["training"]["epochs"]
    total_epochs = start_epoch + 20
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    
    warmup_epochs = 3
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=1.0) 
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # Combine them
    #lr_scheduler = MultiStepLR(optimizer, milestones=[130, 150], gamma=0.1)
    print(f"Starting training from epoch {start_epoch} to {total_epochs}...")
    for epoch in range(start_epoch, total_epochs + 1):
        loss = train_epoch(model, criterion, train_loader, optimizer, device, weight_dict, epoch, scaler)
        lr_scheduler.step()
        print(f"Epoch {epoch}/{num_epochs} Completed | Avg Loss: {loss:.4f}")

        if epoch % cfg["training"]["checkpoint_freq"] == 0:
            save_path = os.path.join(cfg["training"]["checkpoint_dir"], f"swin_detr_epoch{epoch}.pth")
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
