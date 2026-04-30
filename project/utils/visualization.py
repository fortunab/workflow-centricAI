
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_bar_chart(labels, values, title, ylabel, out_path, rotate=False):
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_horizontal_bar(labels, values, title, xlabel, out_path):
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(8, 4.8))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_line_chart(x, y, title, xlabel, ylabel, out_path):
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_text_figure(lines, title, out_path):
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(9, 5.5))
    plt.axis('off')
    plt.title(title)
    y = 0.92
    for line in lines:
        plt.text(0.03, y, line, fontsize=11, family='monospace', va='top')
        y -= 0.09
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
