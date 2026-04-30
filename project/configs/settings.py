from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_DATASET_ID = "SimulaMet/PolypGen-bboxes"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
SAM2_DATASET_ID = "SimulaMet/PolypGen-segmentation"
QWEN_MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct"
QWEN_DATASET_ID = "SimulaMet/Kvasir-VQA-x1"
