import argparse
import yaml
import torch
import json
from torch.utils.data import DataLoader

from src.Dataset.Dataset import BDDDetectionDataset
from src.Models.Detector_head import make_model


def collate_fn(batch):
    return tuple(zip(*batch))


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="val_predictions.json")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = BDDDetectionDataset(
        cfg["data"]["images"]["val"],
        cfg["data"]["annotations"]["val"]
    )

    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = make_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_predictions = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, tgt in zip(outputs, targets):
                image_id = int(tgt["image_id"])

                boxes = pred["boxes"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": xyxy_to_xywh(box.tolist()),
                        "score": float(score)
                    })

    with open(args.out, "w") as f:
        json.dump(all_predictions, f)

    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
