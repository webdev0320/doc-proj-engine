import json
import torch
import os
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset

# 1. Configuration
MODEL_NAME = "naver-clova-ix/donut-base"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

# 2. Processor and Model
print(f"Loading processor and model: {MODEL_NAME}")
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Drastically reduce image size to prevent Out-Of-Memory silent crashes on CPU/4GB GPUs
processor.image_processor.size = {"height": 1280, "width": 960}

# 3. Data Preparation
print(f"Loading dataset from {DATA_DIR}")
try:
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")
except Exception as e:
    print(f"Failed to load dataset: {e}. Please run prepare_dataset.py first.")
    exit(1)

def json2token(obj):
    if type(obj) == dict:
        result = ""
        for k, v in obj.items():
            result += f"<s_{k}>" + json2token(v) + f"</s_{k}>"
        return result
    elif type(obj) == list:
        return "".join([json2token(item) for item in obj])
    else:
        return str(obj)

# Collect all unique keys from the dataset to add as special tokens
unique_keys = set()
for item in dataset:
    gt_dict = json.loads(item["ground_truth"])["gt_parse"]
    for k in gt_dict.keys():
        unique_keys.add(k)

task_prompt = "<s_cord-v2>"
new_special_tokens = [task_prompt]
for k in unique_keys:
    new_special_tokens.extend([f"<s_{k}>", f"</s_{k}>"])

print(f"Adding {len(new_special_tokens)} special tokens to tokenizer.")
processor.tokenizer.add_tokens(new_special_tokens)
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([task_prompt])[0]
model.decoder.resize_token_embeddings(len(processor.tokenizer))

class DonutDataset(Dataset):
    def __init__(self, dataset, processor, task_prompt):
        self.dataset = dataset
        self.processor = processor
        self.task_prompt = task_prompt
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth = item["ground_truth"]
        
        # Parse ground truth to construct string
        gt_dict = json.loads(ground_truth)
        gt_text = self.task_prompt + json2token(gt_dict["gt_parse"]) + self.processor.tokenizer.eos_token
        
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            gt_text, add_special_tokens=False, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze()
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

train_dataset = DonutDataset(dataset, processor, task_prompt)

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # 1 for 4GB VRAM
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    num_train_epochs=5,
    save_strategy="epoch",
    eval_strategy="no",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=1,
)

# 5. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training complete!")
