from project.training.yolo_detection import summarize_yolo_setup
from project.configs.settings import OUTPUTS_DIR

def main():
    summarize_yolo_setup("outputs/yolo_dataset/data.yaml", OUTPUTS_DIR / "yolo")

if __name__ == "__main__":
    main()
