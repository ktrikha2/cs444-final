import os
import json
import argparse
import contextlib
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_model_fast(pred_file, gt_file, output_dir="evaluation_results"):
    """
    Fast evaluation using pycocotools C-optimized backend.
    Does not require images, only JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("OBJECT DETECTION EVALUATION (pycocotools backend)")
    print("="*60)
    
    print(f"1. Loading Ground Truth from: {gt_file}")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_file)
        
    print(f"   Loading Predictions from: {pred_file}")
    try:
        coco_dt = coco_gt.loadRes(pred_file)
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

    # EVAL
    print("\n2. Evaluating...")
    # 'bbox' is the task type
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
    
    coco_eval.summarize()
    
    # Stats indices: 0=mAP(0.5:0.95), 1=mAP@0.5, 2=mAP@0.75
    stats = coco_eval.stats
    mAP_50 = stats[1]
    mAP_75 = stats[2]
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"mAP@0.5:  {mAP_50:.4f}")
    print(f"mAP@0.75: {mAP_75:.4f}")
    
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}
    
    precisions = coco_eval.eval['precision']
    
    per_cat_ap_50 = {}
    per_cat_ap_75 = {}

    if precisions is not None:
        for k_idx, cat_id in enumerate(coco_gt.getCatIds()):
            #AP @ 0.50 (Index 0 in IoU dim)
            p_50 = precisions[0, :, k_idx, 0, 2]
            valid_p_50 = p_50[p_50 > -1]
            ap_50 = valid_p_50.mean() if len(valid_p_50) > 0 else 0.0
            per_cat_ap_50[cat_names[cat_id]] = ap_50
            
            #AP @ 0.75 (Index 5 in IoU dim)
            p_75 = precisions[5, :, k_idx, 0, 2]
            valid_p_75 = p_75[p_75 > -1]
            ap_75 = valid_p_75.mean() if len(valid_p_75) > 0 else 0.0
            per_cat_ap_75[cat_names[cat_id]] = ap_75

    # Print Report
    print(f"\n{'='*60}")
    print(f"Per-Category AP@0.5:")
    print(f"{'='*60}")
    for name, ap in sorted(per_cat_ap_50.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:15s}: {ap:.4f}")

    results = {
        "mAP@0.5": float(mAP_50),
        "mAP@0.75": float(mAP_75),
        "per_category_AP@0.5": per_cat_ap_50,
        "per_category_AP@0.75": per_cat_ap_75
    }
    
    results_file = os.path.join(output_dir, "evaluation_results_fast.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed results to {results_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Object Detection Results")
    

    parser.add_argument("--pred_file", required=True, help="Path to the predictions JSON file (e.g. results/swin_detr.json)")
    

    parser.add_argument("--gt_file", required=True, help="Path to the COCO Ground Truth JSON file")
    

    parser.add_argument("--output_dir", default="evaluation_results", help="Folder to save the result JSON")

    args = parser.parse_args()

    # Run evaluation
    evaluate_model_fast(args.pred_file, args.gt_file, args.output_dir)