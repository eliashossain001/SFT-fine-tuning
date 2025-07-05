import json
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score
from unsloth import FastLanguageModel

# Download required NLTK data
nltk.download('punkt')

# Correct prompt format used in training (General-purpose SFT)
dpo_prompt = """<|system|>
You are a helpful and intelligent assistant.

<|user|>
{instruction}
{input}

<|assistant|>"""


# (run_id, pretrained_model_name, hf_dataset_name)
# (run_id, pretrained_model_name, hf_dataset_name)
configs = [
    ("run_Dummy_Qwen2.5_3B_SFT", "unsloth/Qwen2.5-3B-Instruct", "EliasHossain/dummy-sft-5000")
]

def find_latest_checkpoint(outputs_dir: Path) -> Path:
    ckpts = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* folders found in {outputs_dir}")
    def idx(p): 
        try:
            return int(p.name.split("-", 1)[1])
        except:
            return -1
    return sorted(ckpts, key=idx)[-1]

def compute_metrics(predictions, references):
    rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = [rouge.score(r, p) for r, p in zip(references, predictions)]
    rouge1 = np.mean([s['rouge1'].fmeasure for s in scores])
    rouge2 = np.mean([s['rouge2'].fmeasure for s in scores])
    rougeL = np.mean([s['rougeL'].fmeasure for s in scores])

    def trad_f1(p, r):
        pt = set(word_tokenize(p.lower())); rt = set(word_tokenize(r.lower()))
        tp = len(pt & rt); fp = len(pt - rt); fn = len(rt - pt)
        prec = tp/(tp+fp) if tp+fp>0 else 0.0
        rec  = tp/(tp+fn) if tp+fn>0 else 0.0
        return 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0

    trads = [trad_f1(p, r) for p, r in zip(predictions, references)]
    trad_f1 = np.mean(trads)

    P, R, F1 = score(predictions, references, lang='en')
    return {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL,
        'traditional_f1': trad_f1,
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }

def evaluate_checkpoint(ckpt_dir: Path, test_data, save_path: Path):
    print(f"→ Loading checkpoint from {ckpt_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use FastLanguageModel to load merged checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(str(ckpt_dir))
    FastLanguageModel.for_inference(model)
    model.to(device).eval()

    preds, refs, records = [], [], []
    for item in tqdm(test_data, desc="Generating"):
        input_text = item['input'].strip() if item['input'].strip() else ""
        prompt = dpo_prompt.format(item['instruction'], input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        text = text.replace("<|assistant|>", "").strip()
        preds.append(text)
        refs.append(item['output'])
        records.append({
            'instruction': item['instruction'],
            'input': item['input'],
            'expected_output': item['output'],
            'model_output': text
        })

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(save_path.parent / "file_name.csv", index=False)

    metrics = compute_metrics(preds, refs)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

if __name__ == "__main__":
    base_path = Path("PUT THE BASE PATH")
    outputs  = base_path / "outputs"

    for run_id, _, dataset_name in configs:
        print(f"\n=== Processing {run_id} ===")
        ds = load_dataset(dataset_name)
        split = ds["train"].train_test_split(test_size=0.15, seed=42)
        test = split["test"]

        try:
            ckpt = find_latest_checkpoint(outputs)
        except FileNotFoundError as e:
            print(str(e))
            continue

        save_file = base_path / "evaluation" / f"{run_id}_final_results.json"
        results = evaluate_checkpoint(ckpt, test, save_file)
        print(f"✓ Saved metrics to {save_file}")
        print(json.dumps(results, indent=2))
