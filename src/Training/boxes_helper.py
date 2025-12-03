import torch


def box_xywh_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 4] in absolute [x, y, w, h]
    returns: [N, 4] in [cx, cy, w, h] (still absolute)
    """
    x_min, y_min, w, h = x.unbind(-1)
    cx = x_min + 0.5 * w
    cy = y_min + 0.5 * h
    return torch.stack((cx, cy, w, h), dim=-1)


def box_cxcywh_to_xywh(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 4] in [cx, cy, w, h]
    returns: [N, 4] in [x, y, w, h]
    """
    cx, cy, w, h = x.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    return torch.stack((x_min, y_min, w, h), dim=-1)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, 4] in [cx, cy, w, h]
    returns: [N, 4] in [x_min, y_min, x_max, y_max]
    """
    cx, cy, w, h = x.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU between two sets of boxes.
    boxes1: [N, 4] in xyxy
    boxes2: [M, 4] in xyxy
    returns: [N, M]
    """
    # intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # union
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    # smallest enclosing box
    x1_c = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    y1_c = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    x2_c = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    y2_c = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    enclose_w = (x2_c - x1_c).clamp(min=0)
    enclose_h = (y2_c - y1_c).clamp(min=0)
    enclose = enclose_w * enclose_h

    giou = iou - (enclose - union) / enclose.clamp(min=1e-6)
    return giou
