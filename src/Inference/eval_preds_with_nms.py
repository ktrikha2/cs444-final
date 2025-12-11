import os
import sys
import argparse
import yaml
import json

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms  # Ensure this is imported

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from src.Dataset.Dataset import BDDDetectionDataset
from src.Models.CNN_Swin_Detr import build_swin_detr
from src.Training.boxes_helper import box_cxcywh_to_xywh
from src.Dataset.Transform import get_val_transforms

from torch.utils.data import Subset


def collate_fn(batch):
    # (list_of_images, list_of_targets)
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="val_predictions_swin_detr.json")
    parser.add_argument("--score_thresh", type=float, default=0.05) # Lowered default to let NMS decide
    parser.add_argument("--nms_thresh", type=float, default=0.5)    # New argument for NMS
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
    val_ds_f = BDDDetectionDataset(
        cfg["data"]["images"]["val"],
        cfg["data"]["annotations"]["val"],
        transforms=get_val_transforms(),
    )

    # ---- USE ONLY FIRST 7K IMAGES ----
    subset_size = 1000
    val_ds = Subset(val_ds_f, list(range(min(subset_size, len(val_ds_f)))))
    # --------

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
    model.eval()

    num_classes = cfg["model"]["num_classes"]
    all_predictions = []

    print(f"Starting inference with Score Thresh={args.score_thresh} and NMS IoU={args.nms_thresh}...")

    # -------------------------
    # Inference loop
    # -------------------------
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            batch_tensor = torch.stack(images, dim=0)

            outputs = model(batch_tensor)
            pred_logits = outputs["pred_logits"]
            pred_boxes  = outputs["pred_boxes"]

            B, Q, _ = pred_logits.shape

            for b in range(B):
                image_id = int(targets[b]["image_id"])

                # Probabilities
                probs = pred_logits[b].softmax(-1)
                scores, labels = probs.max(-1)

                # Filter out "no-object" (background)
                keep = labels != num_classes
                scores = scores[keep]
                labels = labels[keep]
                boxes  = pred_boxes[b][keep]

                if boxes.numel() == 0:
                    continue

                # Filter by low score threshold BEFORE NMS to save computation
                # (But keep it low enough so NMS has candidates to work with)
                high_conf = scores > args.score_thresh
                scores = scores[high_conf]
                labels = labels[high_conf]
                boxes = boxes[high_conf]

                if boxes.numel() == 0:
                    continue

                # -----------------------------------------------------------
                # NMS PREPARATION
                # -----------------------------------------------------------
                img_h, img_w = images[b].shape[-2:]
                
                # Convert normalized CXCYWH -> Absolute XYXY for NMS
                cx, cy, w_norm, h_norm = boxes.unbind(-1)
                x1 = (cx - 0.5 * w_norm) * img_w
                y1 = (cy - 0.5 * h_norm) * img_h
                x2 = (cx + 0.5 * w_norm) * img_w
                y2 = (cy + 0.5 * h_norm) * img_h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # -----------------------------------------------------------
                # APPLY NMS PER CLASS
                # -----------------------------------------------------------
                final_boxes_xywh = []
                final_scores = []
                final_labels = []

                unique_labels = labels.unique()
                for lbl in unique_labels:
                    # Get indices for this class
                    cls_indices = (labels == lbl).nonzero(as_tuple=True)[0]
                    
                    cls_boxes_xyxy = boxes_xyxy[cls_indices]
                    cls_scores = scores[cls_indices]

                    # Apply NMS
                    keep_indices = nms(cls_boxes_xyxy, cls_scores, iou_threshold=args.nms_thresh)

                    # Gather kept elements
                    nms_indices = cls_indices[keep_indices]
                    
                    # Convert the kept boxes back to XYWH for JSON output
                    # The original boxes tensor was normalized CXCYWH. 
                    # We need Absolute XYWH.
                    
                    # Get the normalized boxes that passed NMS
                    kept_norm_boxes = boxes[nms_indices] 
                    
                    # Convert normalized cxcywh -> absolute xywh
                    # scale tensor
                    scale = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
                    abs_xywh = box_cxcywh_to_xywh(kept_norm_boxes) * scale
                    
                    final_boxes_xywh.append(abs_xywh)
                    final_scores.append(scores[nms_indices])
                    final_labels.append(labels[nms_indices])

                if len(final_boxes_xywh) == 0:
                    continue

                # Concatenate results from all classes
                final_boxes_xywh = torch.cat(final_boxes_xywh, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                final_labels = torch.cat(final_labels, dim=0)

                # -----------------------------------------------------------
                # STORE PREDICTIONS
                # -----------------------------------------------------------
                for box, score, label in zip(final_boxes_xywh, final_scores, final_labels):
                    # labels in training 0..9 -> category_id 1..10
                    category_id = int(label.item() + 1)
                    x, y, w, h = box.tolist()
                    
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score.item()),
                    })

    with open(args.out, "w") as f:
        json.dump(all_predictions, f)

    print(f"Saved {len(all_predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()