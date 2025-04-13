from datasets import load_dataset

# Load original GSM8K dataset
gsm8k = load_dataset("openai/gsm8k", "main")  # splits: train, test

import re

# Helper: Extract final numeric answer after "####"
def extract_final_answer(answer_text):
    match = re.search(r"####\s*([^\n]+)", answer_text)
    return match.group(1).strip() if match else "N/A"

# Format example into prompt/completion
def format_instruction(example, include_reasoning=True):
    question = example['question']
    if include_reasoning:
        answer = example['answer']
    else:
        answer = extract_final_answer(example['answer'])
    return {
        "prompt": f"Q: {question}\nA: ",
        "completion": f"{answer}"
    }

# Process and save a version of the dataset
def process_and_export(gsm8k, include_reasoning=True, output_prefix="gsm8k"):
    version = "reasoning" if include_reasoning else "numerical"
    for split in gsm8k:
        # Map to only prompt/completion
        processed = gsm8k[split].map(lambda ex: format_instruction(ex, include_reasoning=include_reasoning))
        processed = processed.remove_columns([col for col in processed.column_names if col not in ["prompt", "completion"]])
        # Save to JSONL
        output_file = f"{output_prefix}_{version}_{split}.jsonl"
        processed.to_json(output_file, lines=True)
        print(f"Saved: {output_file}")

# Export both versions
process_and_export(gsm8k, include_reasoning=True, output_prefix="gsm8k")
process_and_export(gsm8k, include_reasoning=False, output_prefix="gsm8k")
