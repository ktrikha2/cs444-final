import os
import json
import contextlib
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ============================================
# MAIN EVALUATION WRAPPER
# ============================================

def evaluate_model_fast(pred_file, gt_file, output_dir="evaluation_results"):
    """
    Fast evaluation using pycocotools C-optimized backend.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FAST OBJECT DETECTION EVALUATION (pycocotools)")
    print("="*60)
    
    # 1. Load Ground Truth
    # Suppress print output from COCO init to keep logs clean
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_file)
        
    print(f"Loaded Ground Truth: {len(coco_gt.getImgIds())} images")

    # 2. Load Predictions
    try:
        # loadRes expects a list of dictionaries: [{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}]
        coco_dt = coco_gt.loadRes(pred_file)
        print(f"Loaded Predictions: {len(coco_dt.getAnnIds())} detections")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

    # 3. Initialize COCOeval
    # 'bbox' is the task type
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # 4. Run Evaluation
    print("\nEvaluating...")
    coco_eval.evaluate()
    print("Accumulating...")
    coco_eval.accumulate()
    print("Summarizing...")
    coco_eval.summarize()

    # ============================================
    # EXTRACT METRICS
    # ============================================
    
    # COCOeval stats array indices:
    # 0: AP @ IoU=0.50:0.95 (mAP)
    # 1: AP @ IoU=0.50
    # 2: AP @ IoU=0.75
    stats = coco_eval.stats
    
    mAP_50 = stats[1]
    mAP_75 = stats[2]
    mAP_overall = stats[0]

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"mAP (0.50:0.95): {mAP_overall:.4f}")
    print(f"mAP@0.50:        {mAP_50:.4f}")
    print(f"mAP@0.75:        {mAP_75:.4f}")

    # ============================================
    # PER-CATEGORY AP (Replicating your old logic)
    # ============================================
    
    print(f"\n{'='*60}")
    print(f"Per-Category AP@0.50:")
    print(f"{'='*60}")

    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}
    
    per_cat_ap = {}

    # precision array dims: [TxRxKxAxM]
    # T=iou thresholds (0 to 9, index 0 is 0.5, index 2 is 0.75)
    # R=recall thresholds
    # K=categories
    # A=area ranges
    # M=max dets
    
    precisions = coco_eval.eval['precision']
    
    # Check if we have valid precision data
    if precisions is not None:
        for k_idx, cat_id in enumerate(coco_gt.getCatIds()):
            # Get precision for IoU=0.50 (index 0), all recalls, specific category, all areas, max dets
            # Mean across recall thresholds (-1 means invalid)
            p = precisions[0, :, k_idx, 0, 2] # 2 is maxDets=100 usually
            
            # Filter out -1 values (which mean no ground truth for that recall bin)
            valid_p = p[p > -1]
            
            if len(valid_p) > 0:
                ap = valid_p.mean()
            else:
                ap = 0.0
            
            per_cat_ap[cat_names[cat_id]] = ap
            
        # Print sorted by AP
        for name, ap in sorted(per_cat_ap.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:15s}: {ap:.4f}")

    # ============================================
    # SAVE JSON
    # ============================================
    results = {
        "mAP": float(mAP_overall),
        "mAP@0.5": float(mAP_50),
        "mAP@0.75": float(mAP_75),
        "per_category_AP@0.5": per_cat_ap
    }
    
    results_file = os.path.join(output_dir, "evaluation_results_fast.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_file}")
    
    return results

if __name__ == "__main__":
    # Configure paths
    PRED_FILE = "resultsdebug/swin_detr_10.json" 
    GT_FILE = "/work/nvme/bfdu/ktrikha/code/cs444-final/Data/det_val_10k.json"
    OUTPUT_DIR = "evaluation_results_debug_full"
    
    # Run evaluation
    evaluate_model_fast(PRED_FILE, GT_FILE, OUTPUT_DIR)