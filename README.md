# SFT Fine-Tuning Pipeline

This repository provides a complete end-to-end **Supervised Fine-Tuning (SFT)** pipeline for various open-source LLMs, including **Qwen2.5** and **LLaMA3.2**, on CWE and security datasets.

---

## 📁 Folder Structure

```
sft-clean/
├── configs/                # Generate configuration files
│   └── generate_sft_configs.py
│
├── training/               # Model-specific training scripts
│   ├── train_cwe_llama3p2_1b_sft.py
│   ├── train_cwe_llama3p2_3b_sft.py
│   ├── train_cwe_qwen2p5_0p5b_sft.py
│   ├── train_cwe_qwen2p5_1p5b_sft.py
│   └── train_cwe_qwen2p5_3b_sft.py
│
├── evaluation/             # Evaluation logic and metrics
│   └── evaluate_sft_results.py
│
├── deployment/             # Push models to Hugging Face
│   └── deploy_to_huggingface.py
│
├── pipeline/               # End-to-end fine-tuning orchestration
│   └── sft_finetuner_pipeline.py
│
├── requirements.txt        # Project dependencies
└── README.md               # You're here
```

---

## Features

-  Support for multiple LLMs (Qwen2.5, LLaMA3.2)
-  Modular design with separate training, evaluation, and deployment
-  Config generator for fast experimentation
-  Hugging Face model hub deployment
-  LoRA + PEFT integration ready

---

##  Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/sft-clean.git
cd sft-clean

# (Optional) Create and activate virtual env
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Training Example

```bash
python training/train_cwe_qwen2p5_3b_sft.py
```

##  Evaluation Example

```bash
python evaluation/evaluate_sft_results.py
```

##  Deployment Example

```bash
python deployment/deploy_to_huggingface.py
```

---

##  Environment Variables (for secrets)

Use a `.env` file or export environment variables before running:

```bash
export HF_TOKEN="your_huggingface_token"
```

---

##  Contributing

Feel free to fork, improve, or raise issues!

---

##  License

MIT License
