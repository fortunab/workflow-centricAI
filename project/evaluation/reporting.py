from pathlib import Path
from project.utils.io import write_text_report, write_json, ensure_dir

def save_pipeline_summary(payload, out_dir):
    out_dir = ensure_dir(out_dir)
    serializable = {
        "classification": getattr(payload["classification"], "__dict__", str(payload["classification"])),
        "num_boxes": len(payload["detections"].boxes),
        "num_masks": len(payload["segmentations"].masks),
        "reasoning": payload["reasoning"],
    }
    write_json(Path(out_dir) / "pipeline_summary.json", serializable)
    write_text_report(Path(out_dir) / "pipeline_summary.txt", "Medical Orchestrator Summary", [("Payload", str(serializable))])
    return serializable
