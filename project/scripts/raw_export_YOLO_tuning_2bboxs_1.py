# ===== CELL 0 =====
!pip install -q accelerate datasets timm opencv-python pillow


# ===== CELL 1 =====
!pip uninstall -y transformers accelerate peft
!pip install transformers==4.44.2 accelerate>=0.26.0


# ===== CELL 2 =====
!pip install ultralytics


# ===== CELL 3 =====
import os
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from ultralytics import YOLO

data_dir = Path("yolo_dataset")
for split in ["train", "val"]:
    (data_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / split / "labels").mkdir(parents=True, exist_ok=True)

full_ds = load_dataset("halyusuf/PolypGen2.0", split="train")
split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
datasets = {"train": split_ds["train"], "val": split_ds["test"]}

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    # box is [xmin, ymin, xmax, ymax]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

for split, ds in datasets.items():
    for i, item in enumerate(tqdm(ds, desc=f"Processing {split}")):
        img = item["image"].convert("RGB")
        img_filename = f"{split}_{i}.jpg"
        img_path = data_dir / split / "images" / img_filename
        img.save(img_path)

        # Handle BBox List-of-Lists
        w, h = item["width"], item["height"]
        bboxes = item["objects"]["bbox"] # This is [[x,y,x,y], ...]

        label_path = data_dir / split / "labels" / f"{split}_{i}.txt"
        with open(label_path, "w") as f:
            for box in bboxes:
                if len(box) == 4:
                    yolo_box = convert_to_yolo((w, h), box)
                    f.write(f"0 {' '.join([f'{x:.6f}' for x in yolo_box])}\n")

# Create data.yaml
yaml_content = {
    "path": str(data_dir.absolute()),
    "train": "train/images",
    "val": "val/images",
    "names": {0: "polyp"}
}

with open("data.yaml", "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

# train YOLOv8
model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else "cpu"
)



# ===== CELL 4 =====
import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import os

if os.path.exists('runs/detect/train/weights/best.pt'):
    shutil.copy('runs/detect/train/weights/best.pt', 'polyp_yolo_best.pt')
    print("Model saved as polyp_yolo_best.pt")

model = YOLO('polyp_yolo_best.pt')

test_sample = split_ds["test"][5]
test_image = test_sample["image"].convert("RGB")

results = model.predict(source=test_image, conf=0.25) # conf=0.25

# results[0].plot() returns a BGR numpy array (OpenCV format)
res_plotted = results[0].plot()
res_rgb = res_plotted[:, :, ::-1] # Convert BGR to RGB for Matplotlib

plt.figure(figsize=(10, 10))
plt.imshow(res_rgb)
plt.title("YOLOv8 Polyp Detection Inference")
plt.axis("off")
plt.show()

# Print box details
for box in results[0].boxes:
    print(f"Detected: {model.names[int(box.cls)]} | Confidence: {box.conf.item():.2f} | BBox: {box.xyxy.tolist()}")



# ===== CELL 5 =====
import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os

# 1. Load Model
model_path = 'polyp_yolo_best.pt'
if not os.path.exists(model_path) and os.path.exists('runs/detect/train/weights/best.pt'):
    shutil.copy('runs/detect/train/weights/best.pt', model_path)

model = YOLO(model_path)

# 2. Get Test Sample
# Note: your code used split_ds["test"][1]
test_sample = split_ds["test"][5]
test_image = test_sample["image"].convert("RGB")
gt_bboxes = test_sample["objects"]["bbox"] # Format: [[xmin, ymin, xmax, ymax], ...]

# 3. Run Inference
results = model.predict(source=test_image, conf=0.25)
res_plotted = results[0].plot()
res_rgb = res_plotted[:, :, ::-1] # BGR to RGB

# 4. Visualization Side-by-Side
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# --- Left: Ground Truth ---
ax[0].imshow(test_image)
for box in gt_bboxes:
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=3, edgecolor='lime', facecolor='none'
    )
    ax[0].add_patch(rect)
    ax[0].text(xmin, ymin-10, "GT: polyp", color='lime', weight='bold', fontsize=12)

ax[0].set_title("Ground Truth (Manual Overlay)", fontsize=15)
ax[0].axis("off")

ax[1].imshow(res_rgb)
ax[1].set_title("YOLOv8 Prediction", fontsize=15)
ax[1].axis("off")

plt.tight_layout()
plt.show()

print(f"Ground Truth Boxes: {gt_bboxes}")
for box in results[0].boxes:
    print(f"Predicted: {model.names[int(box.cls)]} | Conf: {box.conf.item():.2f} | BBox: {box.xyxy.tolist()}")

