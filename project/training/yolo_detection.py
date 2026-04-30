from pathlib import Path
from project.utils.io import ensure_dir, write_text_report, write_json

def convert_bbox_to_yolo(bbox, image_w, image_h):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_w
    y_center = (y_min + height / 2) / image_h
    width /= image_w
    height /= image_h
    return x_center, y_center, width, height

def summarize_yolo_setup(data_yaml, output_dir):
    metrics = {"data_yaml": str(data_yaml), "status": "dataset prepared"}
    write_json(Path(output_dir) / "yolo_setup.json", metrics)
    write_text_report(Path(output_dir) / "yolo_setup.txt", "YOLO Setup", [("Summary", str(metrics))])
    return metrics
