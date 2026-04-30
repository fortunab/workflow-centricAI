# ===== CELL 1 =====
!pip install --upgrade -qqq uv
!uv pip install -qqq torch==2.8.0 torchvision bitsandbytes xformers==0.0.32.post2 \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
!uv pip install --upgrade --no-deps tokenizers trl==0.22.2 transformers==5.2.0

!uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0
!pip install pillow datasets


# ===== CELL 2 =====
import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Load with 4bit to save VRAM, but use Float32 for compute stability on T4
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-0.8B",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)


# ===== CELL 3 =====
# Apply LoRA
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    random_state=3407,
)


# ===== CELL 4 =====
# We must ensure NO BFloat16 tensors exist.
FastVisionModel.for_training(model)
model = model.to(torch.float32) # Force base to Float32



# ===== CELL 5 =====
# dataset
dataset = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train").select(range(1000))
def formatting_prompts_func(examples):
    conversations = []
    for q, a, img in zip(examples["question"], examples["answer"], examples["image"]):
        conv = [
            {"role": "user", "content": [{"type": "text", "text": q}, {"type": "image", "image": img}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        conversations.append(conv)
    return {"messages": conversations}
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)



# ===== CELL 6 =====
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    max_seq_length=1024,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        optim="adamw_8bit",
        logging_steps=1,
        output_dir="outputs",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}
    ),
)


# ===== CELL 7 =====

for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

trainer.train()


# ===== CELL 8 =====
# LoRA, light
model.save_pretrained("qwen3_5_kvasir_lora")
tokenizer.save_pretrained("qwen3_5_kvasir_lora")


# ===== CELL 9 =====
# Merge LoRA into the main model and save as 16-bit (Float16)
# This is much faster for future inference
model.save_pretrained_merged("qwen3_5_kvasir_merged", tokenizer, save_method = "merged_16bit")



# ===== CELL 10 =====
from PIL import Image
from transformers import TextStreamer

from unsloth import FastVisionModel

# Load the merged model
model, tokenizer = FastVisionModel.from_pretrained(
    "qwen3_5_kvasir_merged",
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)

test_sample = dataset[0]
# If 'dataset' was mapped, 'messages' contains the data.
# Let's pull the first user message:
image = test_sample["messages"][0]["content"][1]["image"]
question = test_sample["messages"][0]["content"][0]["text"]

FastVisionModel.for_inference(model)

instruction = [
    {"role": "user", "content": [
        #{"type": "text", "text": f"Analyze this endoscopy image: {question}"},
        {"type": "text", "text": f"{question}"},
        {"type": "image", "image": image}
    ]}
]

input_text = tokenizer.apply_chat_template(instruction, add_generation_prompt=True)
inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)

print(f"\n--- Question: {question} ---")
print("--- Model Answer: ---")
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
    temperature=0.1, # Low temperature for medical accuracy
)


# ===== CELL 11 =====
!pip install -qqq evaluate rouge_score sacrebleu tqdm

import torch
import gc
import evaluate
from unsloth import FastVisionModel
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Setup (first 10 rows)
eval_dataset = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train").select(range(10))

def get_model_predictions(model_name, dataset):
    """Loads a model with higher sequence limits and runs inference."""
    print(f"\n--- Loading Model: {model_name} ---")

    # increase max_seq_length to 4096 to prevent image token truncation
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        max_seq_length=4096,
    )
    FastVisionModel.for_inference(model)

    predictions = []
    references = []

    print(f"Running inference on {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        image = sample["image"]
        question = sample["question"]
        reference = sample["answer"]

        instruction = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image", "image": image}
            ]}
        ]

        input_text = tokenizer.apply_chat_template(instruction, add_generation_prompt=True)

        # Limit pixels to ensure the token count stays within the 4096 limit
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            min_pixels=224*224,
            max_pixels=448*448,
        ).to("cuda")

        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    use_cache=True,
                    temperature=0.1,
                )

            # Extract only the assistant's response
            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            predictions.append(pred_text)
            references.append(reference)

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            predictions.append("")
            references.append(reference)

    # Clean up VRAM
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return predictions, references


# Base model
base_preds, references = get_model_predictions("unsloth/Qwen3.5-0.8B", eval_dataset)

# Fine-tuned model
ft_preds, _ = get_model_predictions("qwen3_5_kvasir_merged", eval_dataset)

def calculate_metrics(preds, refs):
    valid_indices = [i for i, p in enumerate(preds) if p != ""]
    filtered_preds = [preds[i] for i in valid_indices]
    filtered_refs = [refs[i] for i in valid_indices]

    rouge_results = rouge.compute(predictions=filtered_preds, references=filtered_refs)
    bleu_results = bleu.compute(predictions=filtered_preds, references=filtered_refs)

    return {
        "ROUGE-1": rouge_results['rouge1'],
        "ROUGE-L": rouge_results['rougeL'],
        "BLEU": bleu_results['bleu']
    }

base_metrics = calculate_metrics(base_preds, references)
ft_metrics = calculate_metrics(ft_preds, references)

print("\n" + "="*50)
print(f"{'Metric':<15} | {'Base Model':<15} | {'Fine-Tuned':<15}")
print("-" * 50)
for key in base_metrics.keys():
    print(f"{key:<15} | {base_metrics[key]:.4f}{' ':<10} | {ft_metrics[key]:.4f}")
print("="*50)

print("\nSample Output")
print(f"Q: {eval_dataset[0]['question']}")
print(f"Base: {base_preds[0]}")
print(f"FT: {ft_preds[0]}")


# ===== CELL 12 =====
!pip install -qqq nltk


# ===== CELL 13 =====
import evaluate
import numpy as np

# metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor") # Better for capturing synonyms and meaning

def calculate_metrics_improved(preds, refs):
    # Filter out empty predictions
    valid_indices = [i for i, p in enumerate(preds) if p.strip() != ""]
    filtered_preds = [preds[i] for i in valid_indices]
    filtered_refs = [refs[i] for i in valid_indices]

    if not filtered_preds:
        return {"Error": "No valid predictions to score"}

    # 1. ROUGE
    rouge_results = rouge.compute(predictions=filtered_preds, references=filtered_refs)

    # 2. BLEU with Smoothing
    # We use 'smooth=True' to give credit for partial matches (1-grams, 2-grams)
    bleu_results = bleu.compute(predictions=filtered_preds, references=filtered_refs, smooth=True)

    # 3. METEOR
    meteor_results = meteor.compute(predictions=filtered_preds, references=filtered_refs)

    return {
        "ROUGE-L": rouge_results['rougeL'],
        "BLEU (Smoothed)": bleu_results['bleu'],
        "METEOR": meteor_results['meteor']
    }

# Assuming you still have 'base_preds', 'ft_preds', and 'references' in memory:
base_metrics = calculate_metrics_improved(base_preds, references)
ft_metrics = calculate_metrics_improved(ft_preds, references)

print("\n" + "="*60)
print(f"{'Metric':<20} | {'Base Model':<15} | {'Fine-Tuned':<15}")
print("-" * 60)
for key in base_metrics.keys():
    print(f"{key:<20} | {base_metrics[key]:.4f}{' ':<10} | {ft_metrics[key]:.4f}")
print("="*60)


# ===== CELL 15 =====
from datasets import load_dataset

ds = load_dataset("RGarrido03/kvasir-seg-augmented")


# ===== CELL 16 =====
from transformers import pipeline

pipe = pipeline("mask-generation", model="facebook/sam2-hiera-small")


# ===== CELL 17 =====
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/sam2-hiera-small")
model = AutoModel.from_pretrained("facebook/sam2-hiera-small")


# ===== CELL 18 =====
ds


# ===== CELL 19 =====
import torch

pipe = pipeline("mask-generation", model="facebook/sam2-hiera-small")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    pipe.set_image("img_kvasirseg.jpg")
    masks, _, _ = pipe.predict("colorectal polyps")



# ===== CELL 21 =====
!pip install git+https://github.com/facebookresearch/segment-anything-2.git


# ===== CELL 22 =====
import os
import torch

# Instalare SAM 2
!pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Descărcare checkpoint-uri (folosim varianta 'tiny' pentru T4)
!wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt


# ===== CELL 23 =====
import os
import torch

# Instalăm SAM 2 direct din sursă
!pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Descărcăm checkpoint-ul tiny pentru a rămâne în limitele memoriei T4 (la fel ca la Qwen)
!mkdir -p checkpoints
if not os.path.exists("checkpoints/sam2_hiera_tiny.pt"):
    !wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt


# ===== CELL 24 =====
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
