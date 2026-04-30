import numpy as np
from project.utils.io import write_json, write_text_report, ensure_dir

def format_vqa_conversations(examples):
    conversations = []
    for q, a, img in zip(examples["question"], examples["answer"], examples["image"]):
        conversations.append({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": q}, {"type": "image", "image": img}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ]
        })
    return {"conversations": conversations}

def simple_text_metrics(predictions, references):
    exact = np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)]) if references else 0.0
    return {"exact_match": float(exact), "count": len(references)}

def save_reasoning_metrics(metrics, output_dir="outputs/qwen"):
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "qwen_metrics.json", metrics)
    write_text_report(output_dir / "qwen_metrics.txt", "Qwen Evaluation", [("Metrics", str(metrics))])
