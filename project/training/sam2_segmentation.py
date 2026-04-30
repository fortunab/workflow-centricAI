import numpy as np
from project.utils.io import write_text_report, write_json, ensure_dir

def get_bounding_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 1, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def calculate_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    dice = (2 * inter) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    return {"iou": float(iou), "dice": float(dice)}

def evaluate_masks(pred_masks, gt_masks, output_dir):
    output_dir = ensure_dir(output_dir)
    scores = [calculate_metrics(p, g) for p, g in zip(pred_masks, gt_masks)]
    mean_iou = float(np.mean([s["iou"] for s in scores])) if scores else 0.0
    mean_dice = float(np.mean([s["dice"] for s in scores])) if scores else 0.0
    payload = {"mean_iou": mean_iou, "mean_dice": mean_dice, "num_samples": len(scores)}
    write_json(output_dir / "sam2_eval.json", payload)
    write_text_report(output_dir / "sam2_eval.txt", "SAM2 Evaluation", [("Metrics", str(payload))])
    return payload
