#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install -q bitsandbytes accelerate peft transformers')




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, prepare_model_for_kbit_training
import json
import re
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/10623Project')




# 2. Point to base model and adapter directory
BASE_MODEL = "google/gemma-7b"
HF_TOKEN = "token"




# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=HF_TOKEN)



# 4. Configure 4‑bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)




# 5. Load the base 4‑bit model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",          # automatically places layers on available GPUs/CPU
    torch_dtype=torch.bfloat16,
    use_auth_token=HF_TOKEN
)




# 6. Prepare for k‑bit (LoRA) training/inference
model = prepare_model_for_kbit_training(model)




ADAPTER_DIR = "./qlora_gemma_gsm8k/adapter"




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# 7. Load trained LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_DIR, torch_dtype=torch.bfloat16)




model.to(DEVICE)
model.eval()




prompt = (
    "### Instruction:\nSolve the following math problem step by step.\n\n"
    "### Input:\nNatalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\n"
    "### Response:"
)
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)



# Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))




with open("gsm8k_reasoning_test.jsonl", "r") as f:  # change to your actual file path
    test_data = [json.loads(line) for line in f]




batch_size = 16
all_results = []
instruction = "Solve the following math problem step by step."

for start in tqdm(range(0, len(test_data), batch_size)):
# for start in range(8, 16, batch_size):
    batch = test_data[start:start + batch_size]

    prompts = []

    for item in batch:
        # extract just the question text
        q = item["prompt"].split("A:")[0].replace("Q:", "").strip()
        prompt = (
              f"### Instruction:\n{instruction}\n\n"
              f"### Input:\n{q}\n\n"
              "### Response:"
        )
        prompts.append(prompt)

    # tokenize as a batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=384
    ).to(DEVICE)

    # generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode all at once
    answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # pair up and store
    for item, ans in zip(batch, answers):
        all_results.append({
            "original_prompt": item["prompt"],
            "generated_answer": ans
        })




final_numerical_answers = []
for result in tqdm(all_results):
    # print("question: ", result["question"])
    # print("answer: ", result["generated_answer"])
    # print("\n")
    parts = result["generated_answer"].split("####")
    if len(parts) >= 3:
        content_between = parts[1].strip()
        final_numerical_answers.append(content_between)
    else:
        final_numerical_answers.append(None)
print(final_numerical_answers)




ground_truth = []

with open("gsm8k_numerical_test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        ground_truth.append(data["completion"])
print("prediction length: ", len(final_numerical_answers))
print("ground truth length: ", len(ground_truth))
print(ground_truth[:8])




# calculate accuracy
def normalize_answer(ans):
    ans = ans.split(" ")[0]
    ans = ans.replace("%", "").replace("$", "").replace(",", "").strip()
    try:
        return str(int(float(ans)))
    except:
        return ans

count = 0
for i in range(len(final_numerical_answers)):
    prediction = final_numerical_answers[i]
    if prediction:
        label = ground_truth[i]
        if label == normalize_answer(prediction):
            count += 1
print(count / len(final_numerical_answers))




# for three shot learning
batch_size = 16
all_results = []
instruction = "Solve the following math problem step by step."

few_shot_examples = [
    {
        "input": "A car travels at 50 mph for 2 hours and then at 70 mph for 3 hours. What is the total distance traveled by the car?",
        "output": "Distance₁ = 50 × 2 = <<50*2=100>>100 miles.\nDistance₂ = 70 × 3 = <<70*3=210>>210 miles.\nTotal distance = 100 + 210 = <<100+210=310>>310 miles.\n#### 310"
    },
    {
        "input": "To make a lemonade mixture, you mix 3 liters of lemon concentrate with water to produce 12 liters of lemonade. How many liters of water do you need to add?",
        "output": "Total lemonade = 12 liters, lemon concentrate = 3 liters.\nWater needed = 12 − 3 = <<12-3=9>>9 liters.\n#### 9"
    },
    {
        "input": "A jacket originally costs $120. It is marked down by 25%, and then an additional 10% discount is applied to the sale price. What is the final price of the jacket?",
        "output": "First discount: 25% of 120 = <<120*0.25=30>>30 dollars.\nPrice after first markdown = 120 − 30 = <<120-30=90>>90 dollars.\nSecond discount: 10% of 90 = <<90*0.10=9>>9 dollars.\nFinal price = 90 − 9 = <<90-9=81>>81 dollars.\n#### 81"
    },
]

for start in tqdm(range(0, len(test_data), batch_size)):
# for start in range(8, 16, batch_size):
    batch = test_data[start:start + batch_size]

    prompts = []

    for item in batch:
        # extract just the question text
        q = item["prompt"].split("A:")[0].replace("Q:", "").strip()

        # Build the few-shot prefix
        few_shot_str = ""
        for ex in few_shot_examples:
            few_shot_str += f"### Input:\n{ex['input']}\n\n"
            few_shot_str += f"### Response:\n{ex['output']}\n\n"

        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"{few_shot_str}"
            f"### Input:\n{q}\n\n"
            "### Response:\n"
        )
        prompts.append(prompt)

    # tokenize as a batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    # generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode all at once
    answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # pair up and store
    for item, ans in zip(batch, answers):
        all_results.append({
            "original_prompt": item["prompt"],
            "generated_answer": ans
        })




final_numerical_answers = []
for result in tqdm(all_results):
    # print("original_prompt: ", result["original_prompt"])
    # print("answer: ", result["generated_answer"])
    # print("\n")
    parts = result["generated_answer"].split("####")
    if len(parts) >= 5:
        content_between = parts[4].strip()
        final_numerical_answers.append(content_between)
    else:
        final_numerical_answers.append(None)
print(final_numerical_answers)



ground_truth = []

with open("gsm8k_numerical_test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        ground_truth.append(data["completion"])
print("prediction length: ", len(final_numerical_answers))
print("ground truth length: ", len(ground_truth))
print(ground_truth[:8])




# calculate accuracy
def normalize_answer(ans):
    ans = ans.split(" ")[0]
    ans = ans.replace("%", "").replace("$", "").replace(",", "").strip()
    try:
        return str(int(float(ans)))
    except:
        return ans

count = 0
for i in range(len(final_numerical_answers)):
    prediction = final_numerical_answers[i]
    if prediction:
        label = ground_truth[i]
        if label == normalize_answer(prediction):
            count += 1
print(count / len(final_numerical_answers))

