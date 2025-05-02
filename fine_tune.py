#!/usr/bin/env python
# coding: utf-8



from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/10623Project')




# !pip install bitsandbytes datasets
get_ipython().system('pip install -U bitsandbytes')
get_ipython().system('pip install datasets')
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import json




# Step 1: Load and process dataset (assuming JSONL format)
def load_jsonl_dataset(file_path):
    with open(file_path, "r") as f:
        lines = [json.loads(line) for line in f]
    return lines

def format_for_instruction(example):
    instruction = "Solve the following math problem step by step."
    question = example["prompt"].replace("Q:", "").replace("A:", "").strip()
    return {
        "prompt": f"### Instruction:\n{instruction}\n\n### Input:\n{question}\n\n### Response:",
        "completion": example["completion"]
    }


# Load raw and format
data_path = "gsm8k_reasoning_train.jsonl"
raw_data = load_jsonl_dataset(data_path)
formatted_data = [format_for_instruction(ex) for ex in raw_data]
print(formatted_data[0])




# Convert to HF dataset
# !pip install datasets
from datasets import Dataset
train_dataset = Dataset.from_list(formatted_data)

# Tokenization
model_name = "google/gemma-7b"
hf_token = 'tokens'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

def tokenize(example):
    full_input = example["prompt"] + example["completion"]
    return tokenizer(
        full_input,
        truncation=True,
        max_length=384,
        padding="max_length"
    )

tokenized_dataset = train_dataset.map(tokenize, batched=False)




# Step 2: Load 4-bit quantized model with BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    token=hf_token
)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

# Step 3: Apply LoRA (QLoRA = LoRA on quantized model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Check for Gemma-specific names if needed
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)




from transformers import TrainingArguments
# Step 4: Training setup
training_args = TrainingArguments(
    output_dir="./qlora_gemma_gsm8k_reasoning",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    # evaluation_strategy="no",
    eval_strategy='no',
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Step 5: Train
trainer.train()

# Save PEFT adapter
model.save_pretrained("./qlora_gemma_gsm8k/adapter")

