import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from src.Training.boxes_helper import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Computes an assignment between targets and predictions for DETR. Bipartite matching with pred boxes and gt boxes 

    For each batch element, it solves a bipartite matching between
    N_pred queries and N_gt ground truth boxes based on a cost that
    combines class prob, bbox L1, and GIoU.
    """
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
          - "pred_logits": [B, num_queries, num_classes+1]
          - "pred_boxes":  [B, num_queries, 4] (normalized cx,cy,w,h)

        targets: list of len B, each:
          { "labels": [N_i], "boxes": [N_i, 4] in normalized cx,cy,w,h }
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].softmax(-1)  # [B, Q, C+1]
        out_bbox = outputs["pred_boxes"]               # [B, Q, 4]

        indices = []
        for b in range(bs):
            # predicted probabilities and boxes for the batch
            prob = out_prob[b]       # [Q, C+1]
            bbox = out_bbox[b]       # [Q, 4]

            tgt_ids = targets[b]["labels"]     # [N_gt]
            tgt_bbox = targets[b]["boxes"]     # [N_gt, 4] normalized cx,cy,w,h

            if tgt_ids.numel() == 0:
                # no matches
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
                continue

            # classification cost: negative prob of GT class
            # prob[:, tgt_ids] -> [Q, N_gt]
            cost_class = -prob[:, tgt_ids]

            # bbox L1 cost
            cost_bbox = torch.cdist(bbox, tgt_bbox, p=1)  # [Q, N_gt]

            # GIoU cost
            bbox_xyxy = box_cxcywh_to_xyxy(bbox)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(bbox_xyxy, tgt_bbox_xyxy)

            # final cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )  # [Q, N_gt]
            C = C.cpu()

            # Hungarian matching: row idx (pred), col idx (gt)
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )

        return indices
