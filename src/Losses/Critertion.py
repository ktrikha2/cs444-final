# src/Losses/criterion.py
import torch
from torch import nn
import torch.nn.functional as F

from src.Training.boxes_helper import box_xywh_to_cxcywh, box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """
    DETR loss:
      - classification (with "no-object" class)
      - bbox L1
      - GIoU
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        """
        num_classes: number of object classes matcher: HungarianMatcher instance
        weight_dict: dict like {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        eos_coef: weight for the "no-object" class
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        # class weight: [num_classes+1], last is no object/background
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_normalized_targets(self, targets):
        """
        Convert GT boxes from absolute XYWH to normalized CXCYWH in [0,1],
        using per-sample img_size stored in targets[i]["img_size"] = [H, W].
        """
        norm_targets = []
        for t in targets:
            boxes_xywh = t["boxes"]  # absolute
            img_size = t["img_size"]  # [H, W]
            h, w = img_size[0], img_size[1]

            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
            scale = torch.tensor([w, h, w, h], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
            boxes_cxcywh_norm = boxes_cxcywh / scale

            new_t = {
                "labels": t["labels"],
                "boxes": boxes_cxcywh_norm,
            }
            norm_targets.append(new_t)
        return norm_targets

    def _get_src_permutation_idx(self, indices):
        """
        Convert (batch_idx, src_idx) for advanced indexing from matched pairs.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i, dtype=torch.int64) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Cross-entropy loss over classes, including no-object.
        """
        src_logits = outputs["pred_logits"]  # [B, Q, C+1]

        idx = self._get_src_permutation_idx(indices)
        # GT labels for matched queries
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0
        )

        # all queries default to "no-object" (class index = num_classes)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # [B, C+1, Q]
            target_classes,
            self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        L1 and GIoU losses on boxes.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]  # [total_matched, 4]
        target_boxes = torch.cat(
            [t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0
        )

        # L1
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
        loss_giou = (1.0 - giou.diag()).sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        """
        outputs: dict("pred_logits", "pred_boxes")
        targets: list of dict from dataset, each with:
           - "boxes": absolute xywh
           - "labels": class indices
           - "img_size": [H, W] (we'll add this in the train loop)
        returns: dict of individual loss components
        """
        # normalize GT boxes to cx,cy,w,h in [0,1]
        targets_norm = self._get_normalized_targets(targets)

        # compute matching between outputs and normalized GT
        indices = self.matcher(outputs, targets_norm)

        # count number of GT boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        num_boxes = torch.clamp(num_boxes, min=1.0)

        # compute all requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets_norm, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets_norm, indices, num_boxes))
        print("targets[0] boxes:", targets[0]["boxes"][:5])
        print("targets[0] labels:", targets[0]["labels"][:5])
        print("are target boxes normalized?", targets[0]["boxes"].max())
        print("matcher indices:", indices)

        return losses
