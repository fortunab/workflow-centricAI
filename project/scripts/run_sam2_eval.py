import numpy as np
from project.training.sam2_segmentation import evaluate_masks
from project.configs.settings import OUTPUTS_DIR

def main():
    gt = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
    pred = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
    pred[0][8:24, 8:24] = 1
    gt[0][10:22, 10:22] = 1
    evaluate_masks(pred, gt, OUTPUTS_DIR / "sam2")

if __name__ == "__main__":
    main()
