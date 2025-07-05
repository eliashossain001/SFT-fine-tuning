from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import os

# Step 1: Login if not already
login(token="")  # Or use `huggingface-cli login` in terminal

# Step 2: Load local JSON dataset
dataset = load_dataset("json", data_files="dummy_sft_5000.json")

# Step 3: Push to Hugging Face
dataset.push_to_hub("EliasHossain/dummy-sft-5000")
