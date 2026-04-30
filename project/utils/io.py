from pathlib import Path
import json

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_text_report(path, title, sections):
    path = Path(path)
    ensure_dir(path.parent)
    lines = [f"=== {title} ===", ""]
    for header, body in sections:
        lines.append(str(header))
        lines.append(str(body).rstrip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
