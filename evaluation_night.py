import json
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def load_ground_truth(ann_file):
    """Load ground truth from COCO JSON"""
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    # Organize by image_id
    gt_by_image = defaultdict(list)
    for ann in coco['annotations']:
        gt_by_image[ann['image_id']].append({
            'category_id': ann['category_id'],
            'bbox': ann['bbox']
        })
    
    # Load category names
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    # Load image info
    images = {img['id']: img for img in coco['images']}
    
    return gt_by_image, categories, images


def load_predictions(pred_file):
    """Load predictions from JSON"""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # Organize by image_id
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred['image_id']].append({
            'category_id': pred['category_id'],
            'bbox': pred['bbox'],
            'score': pred['score']
        })
    
    return pred_by_image


def box_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] format
    """
    # Convert to [x1, y1, x2, y2]
    box1_x2y2 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2_x2y2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # Calculate intersection
    x1 = max(box1_x2y2[0], box2_x2y2[0])
    y1 = max(box1_x2y2[1], box2_x2y2[1])
    x2 = min(box1_x2y2[2], box2_x2y2[2])
    y2 = min(box1_x2y2[3], box2_x2y2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate Average Precision for a single category
    """
    if len(ground_truths) == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched_gt = set()
    
    for i, pred in enumerate(predictions):
        max_iou = 0
        max_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            
            iou = box_iou(pred['bbox'], gt['bbox'])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Check if match is good enough
        if max_iou >= iou_threshold and max_gt_idx not in matched_gt:
            tp[i] = 1
            matched_gt.add(max_gt_idx)
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    

    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_map(pred_by_image, gt_by_image, categories, iou_threshold=0.5):
    """
    Calculate mean Average Precision across all categories
    """
    from tqdm import tqdm
    
    relevant_image_ids = set(pred_by_image.keys())
    
    aps = {}
    
    print(f"   Calculating AP for {len(categories)} categories on {len(relevant_image_ids)} images...")
    for cat_id in tqdm(categories.keys(), desc="   Categories"):
        # Collect all predictions and ground truths for this category
        cat_predictions = []
        cat_ground_truths = []

        for img_id in relevant_image_ids:
            # Get predictions for this image and category
            img_preds = [p for p in pred_by_image.get(img_id, []) 
                        if p['category_id'] == cat_id]
            cat_predictions.extend(img_preds)
            
            # Get ground truths for this image and category
            img_gts = [gt for gt in gt_by_image.get(img_id, [])
                      if gt['category_id'] == cat_id]
            cat_ground_truths.extend(img_gts)

        if len(cat_ground_truths) > 0:
            ap = calculate_ap(cat_predictions, cat_ground_truths, iou_threshold)
            aps[cat_id] = ap

    if len(aps) == 0:
        return 0.0, aps
    
    mAP = np.mean(list(aps.values()))
    return mAP, aps


def visualize_predictions(image_path, predictions, ground_truths, categories, save_path=None):
    """
    Visualize predictions and ground truth on an image
    """
    img = Image.open(image_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ground Truth
    ax1.imshow(img)
    ax1.set_title("Ground Truth", fontsize=16)
    ax1.axis('off')
    
    for gt in ground_truths:
        x, y, w, h = gt['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor='green', facecolor='none')
        ax1.add_patch(rect)
        cat_name = categories[gt['category_id']]
        ax1.text(x, y-5, cat_name, color='green', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Predictions
    ax2.imshow(img)
    ax2.set_title("Predictions", fontsize=16)
    ax2.axis('off')
    
    for pred in predictions:
        x, y, w, h = pred['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        cat_name = categories[pred['category_id']]
        score = pred['score']
        ax2.text(x, y-5, f"{cat_name} {score:.2f}", color='red', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def evaluate_model(pred_file, gt_file, img_dir, output_dir="evaluation_results", max_images=None):
    """
    Complete evaluation pipeline
    
    Args:
        max_images: If specified, only evaluate on first N images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("OBJECT DETECTION EVALUATION")
    print("="*60)
    
    print("\n1. Loading data...")
    gt_by_image, categories, images = load_ground_truth(gt_file)
    pred_by_image = load_predictions(pred_file)

    if max_images is not None:
        all_image_ids = sorted(list(set(gt_by_image.keys()) & set(pred_by_image.keys())))
        subset_image_ids = all_image_ids[:max_images]
        
        gt_by_image = {k: v for k, v in gt_by_image.items() if k in subset_image_ids}
        pred_by_image = {k: v for k, v in pred_by_image.items() if k in subset_image_ids}
        
        print(f"   LIMITED TO FIRST {max_images} IMAGES")
 
    total_preds = sum(len(preds) for preds in pred_by_image.values())
    total_gts = sum(len(gts) for gts in gt_by_image.values())
    
    print(f"   Ground truth images: {len(gt_by_image)}")
    print(f"   Prediction images: {len(pred_by_image)}")
    print(f"   Total predictions: {total_preds}")
    print(f"   Total ground truths: {total_gts}")
    print(f"   Avg predictions per image: {total_preds / max(len(pred_by_image), 1):.1f}")
    print(f"   Categories: {len(categories)}")
    
    print("\n2. Calculating mAP...")
    mAP_50, aps_50 = calculate_map(pred_by_image, gt_by_image, categories, iou_threshold=0.5)
    mAP_75, aps_75 = calculate_map(pred_by_image, gt_by_image, categories, iou_threshold=0.75)
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"mAP@0.5:  {mAP_50:.4f}")
    print(f"mAP@0.75: {mAP_75:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Per-Category AP@0.5:")
    print(f"{'='*60}")
    for cat_id, ap in sorted(aps_50.items(), key=lambda x: x[1], reverse=True):
        cat_name = categories[cat_id]
        print(f"{cat_name:15s}: {ap:.4f}")

    results = {
        "mAP@0.5": float(mAP_50),
        "mAP@0.75": float(mAP_75),
        "per_category_AP@0.5": {categories[k]: float(v) for k, v in aps_50.items()},
        "per_category_AP@0.75": {categories[k]: float(v) for k, v in aps_75.items()}
    }
    
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed results to {results_file}")
    
    print(f"\n3. Creating visualizations...")
    sample_image_ids = list(pred_by_image.keys())[:5]  
    
    for i, img_id in enumerate(sample_image_ids):
        img_info = images[img_id]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if os.path.exists(img_path):
            save_path = os.path.join(output_dir, f"visualization_{i+1}.png")
            visualize_predictions(
                img_path,
                pred_by_image[img_id],
                gt_by_image[img_id],
                categories,
                save_path
            )
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    PRED_FILE = "/work/nvme/bfdu/dsingh10/output/swin_detr_epoch150_night.json" 
    GT_FILE = "/work/nvme/bfdu/dsingh10/code/cs444-final-fin/Data_Night/det_val_coco.json"
    IMG_DIR = "/work/nvme/bfdu/dsingh10/data_night/Dataset/val"
    OUTPUT_DIR = "dipali_test_night_1000val"

    MAX_IMAGES = None
    
    results = evaluate_model(PRED_FILE, GT_FILE, IMG_DIR, OUTPUT_DIR, max_images=MAX_IMAGES)