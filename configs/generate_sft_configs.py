import os
import re

# 0) Paths — adjust these
script_dir    = os.path.dirname(os.path.abspath(__file__))
base_path     = "PUT THE BASE PATH"  
template_file = os.path.join(script_dir, "sft_finetuning.py")

if not os.path.isfile(template_file):
    raise FileNotFoundError(f"Template not found: {template_file}")

# 1) Your runs
# 1) Elias's Dummy Dataset Runs
configs = [
    ("run_Dummy_Qwen2.5_0.5B_SFT",   "unsloth/Qwen2.5-0.5B-Instruct",         "EliasHossain/dummy-sft-5000"),
    ("run_Dummy_Qwen2.5_1.5B_SFT",   "unsloth/Qwen2.5-1.5B-Instruct",         "EliasHossain/dummy-sft-5000"),
    ("run_Dummy_Qwen2.5_3B_SFT",     "unsloth/Qwen2.5-3B-Instruct",           "EliasHossain/dummy-sft-5000"),
    ("run_Dummy_Llama3.2_1B_SFT",    "unsloth/Llama-3.2-1B-Instruct",         "EliasHossain/dummy-sft-5000"),
    ("run_Dummy_Llama3.2_3B_SFT",    "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit", "EliasHossain/dummy-sft-5000"),
]


# 2) Read the template once
with open(template_file, "r") as f:
    template = f.read()

# 3) For each config, substitute and write
for run_id, model, dataset in configs:
    # make dirs
    run_dir = os.path.join(base_path, run_id)
    os.makedirs(os.path.join(run_dir, "output"),     exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evaluation"), exist_ok=True)

    # do the replacements
    new_content = template
    new_content = re.sub(
        r'run_id\s*=\s*".*"', 
        f'run_id = "{run_id}"', 
        new_content
    )
    new_content = re.sub(
        r'base_model\s*=\s*".*"', 
        f'base_model = "{model}"', 
        new_content
    )
    new_content = re.sub(
        r'dataset_name\s*=\s*".*"', 
        f'dataset_name = "{dataset}"', 
        new_content
    )

    # write it out
    out_path = os.path.join(run_dir, f"{run_id}.py")
    with open(out_path, "w") as out:
        out.write(new_content)

    print(f"✔ Generated {out_path}")
