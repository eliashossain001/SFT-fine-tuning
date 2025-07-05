import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import json

run_id = "run_Dummy_Llama3.2_3B_SFT"
base_model = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
dataset_name = "EliasHossain/dummy-sft-5000"
hf_token=""

# Load and split dataset
dataset = load_dataset(dataset_name)
dataset = dataset.shuffle(seed=42)
split_dataset = dataset["train"].train_test_split(test_size=0.15, seed=42)

# Model initialization
max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
   model_name=base_model,
   max_seq_length=max_seq_length,
   dtype=dtype,
   load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
   model,
   r=64,
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
   lora_alpha=64,
   lora_dropout=0.1,
   bias="none",
   use_gradient_checkpointing="unsloth",
   random_state=3407,
)

# Prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    texts = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

split_dataset["train"] = split_dataset["train"].map(formatting_prompts_func, batched=True)

# Training configuration
trainer = SFTTrainer(
   model=model,
   tokenizer=tokenizer,
   train_dataset=split_dataset["train"],
   dataset_text_field="text",
   max_seq_length=max_seq_length,
   dataset_num_proc=2,
   packing=False,
   args=TrainingArguments(
       per_device_train_batch_size=8,
       gradient_accumulation_steps=8,
       warmup_steps=400,
       max_steps=2500,
       learning_rate=5e-5,
       fp16=False,
       bf16=True,
       logging_steps=50,
       save_steps=700,
       optim="adamw_8bit",
       max_grad_norm=1.0,
       weight_decay=0.05,
       lr_scheduler_type="linear",
       seed=3407,
       output_dir=f"{run_id}/output",
       report_to="none",
   ),
)

# Train and save
trainer.train()
model.push_to_hub(
   f"EliasHossain/{run_id}",
   private=True,
   token=hf_token
)
tokenizer.push_to_hub(
   f"EliasHossain/{run_id}",
   private=True,
   token=hf_token
)