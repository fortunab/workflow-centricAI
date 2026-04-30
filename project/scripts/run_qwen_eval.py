from project.training.qwen_reasoning import simple_text_metrics, save_reasoning_metrics
from project.configs.settings import OUTPUTS_DIR

def main():
    predictions = ["polyp", "no polyp", "uncertain"]
    references = ["polyp", "no polyp", "polyp"]
    metrics = simple_text_metrics(predictions, references)
    save_reasoning_metrics(metrics, OUTPUTS_DIR / "qwen")

if __name__ == "__main__":
    main()
