import os
import sys
import argparse
import yaml
import json

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.Dataset.Dataset import BDDDetectionDataset
from src.Models.CNN_Swin_Detr import build_swin_detr
from src.Training.boxes_helper import box_cxcywh_to_xywh
from src.Dataset.Transform import get_val_transforms
from torchvision.ops import nms


def collate_fn(batch):
    # (list_of_images, list_of_targets)
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="val_predictions_swin_detr.json")
    parser.add_argument("--score_thresh", type=float, default=0.50)
    args = parser.parse_args()

    # -------------------------
    # Load config & device
    # -------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset / DataLoader
    # -------------------------
    val_ds = BDDDetectionDataset(
        cfg["data"]["images"]["val"],
        cfg["data"]["annotations"]["val"],
        transforms=get_val_transforms(),   # no train-time augmentations
    )

    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # -------------------------
    # Build & load model
    # -------------------------
    model = build_swin_detr(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    #missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    #print("MISSING KEYS:", missing)
    #print("UNEXPECTED KEYS:", unexpected)   
    model.eval()
    #model.backbone.train() #um?

    num_classes = cfg["model"]["num_classes"]  # e.g. 10
    all_predictions = []

    # -------------------------
    # Inference loop
    # -------------------------
    printed_debug = False
    with torch.no_grad():
        for images, targets in loader:
            # images: tuple([3,H,W]); targets: tuple(dict)
            images = [img.to(device) for img in images]
            batch_tensor = torch.stack(images, dim=0)  # [B,3,H,W]

            outputs = model(batch_tensor)
            pred_logits = outputs["pred_logits"]  # [B, Q, num_classes+1]
            pred_boxes  = outputs["pred_boxes"]   # [B, Q, 4] normalized cx,cy,w,h

            B, Q, _ = pred_logits.shape

            for b in range(B):
                # image_id from target (dataset should store it)
                image_id = int(targets[b]["image_id"])

                # softmax over classes (+ no-object)
                probs = pred_logits[b].softmax(-1)        # [Q, C+1]
                scores, labels = probs.max(-1)            # [Q], [Q]
                #scores, labels = probs[..., :-1].max(-1)  # ignore no-object class

                if not printed_debug:
                    print("\n===== RAW MODEL DEBUG =====")
                    print("First 10 normalized pred_boxes:")
                    print(pred_boxes[b][:10].cpu())
                    print("First 10 scores:")
                    print(scores[:10].cpu())
                    print("First 10 labels:")
                    print(labels[:10].cpu())
                printed_debug = True
                # ignore "no-object" class, which is index = num_classes
                keep = labels != num_classes
                scores = scores[keep]
                labels = labels[keep]
                boxes  = pred_boxes[b][keep]              # [N,4]

                if boxes.numel() == 0:
                    continue

                # convert from normalized cx,cy,w,h -> absolute xywh
                img_h, img_w = images[b].shape[-2:]
                scale = torch.tensor(
                    [img_w, img_h, img_w, img_h],
                    dtype=boxes.dtype,
                    device=boxes.device,
                )

                boxes_xywh = box_cxcywh_to_xywh(boxes)    # normalized xywh
                boxes_xywh = boxes_xywh * scale           # absolute pixels

                for box, score, label in zip(boxes_xywh, scores, labels):
                    s = float(score.item())
                    if s < args.score_thresh:
                        continue

                    # labels in training are 0..9 → map back to category_id 1..10
                    category_id = int(label.item() + 1)

                    x, y, w, h = box.tolist()
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": s,
                    })

    # -------------------------
    # Save predictions
    # -------------------------
    with open(args.out, "w") as f:
        json.dump(all_predictions, f)

    print(f"Saved {len(all_predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
