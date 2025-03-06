# Sample DeepSeek R1 LLM & WhisperAI Project

This repository contains a **sample project** demonstrating the use of:
- **Whisper LoRA Fine-Tuned** for **Russian automatic speech recognition (ASR)**.
- **DeepSeek Meeting Summary** for **summarizing meeting transcriptions**.

It includes a **small dataset sample**, a **video file for testing**, and **experiments showcasing model fine-tuning and inference.**

---

## 📂 Directory Structure
```
meeting_key_points_ai/
│── sample_datasets/
│   ├── sample_golos/                  # 10 train + 2 test examples from Golos dataset
│   ├── sample_meeting/                # 10 samples from a custom Meeting dataset
│
│── examples/
│   ├── meeting_video.mp4              # Video file for real-world testing
│
│── sample_implementation.ipynb        # Runs models on meeting_video.mp4
│── sample_experiments.ipynb           # Fine-tunes models on sample_datasets
│
│── real_results/
│   ├── notebooks/
│       ├── implementation.ipynb       # Results from sample_implementation.ipynb
│       ├── experiments.ipynb          # Results from sample_experiments.ipynb
│   ├── outputs/                       # CSV reports of experiments
│
│── README.md                          # Project documentation
│── requirements.txt                   # Dependencies
│── .gitignore                         # Excludes unnecessary files
```

---

## 📊 Datasets
### **1. Golos Dataset (subset)**
- **Source:** [Golos GitHub Repository](https://github.com/salute-developers/golos/tree/master/golos#golos-dataset)
- **Contents:** 10 training samples + 2 test samples

### **2. Meeting Dataset (Custom Subset)**
- **Derived from:** **AMI Corpus** & **ICSI Corpus**
- **Contents:** 10 meeting transcription samples

---

## 🎥 Example Video for Testing
The **`examples/meeting_video.mp4`** file is provided to **test the models' transcription and summarization capabilities**.

---

## 🚀 Model Usage
### **Whisper LoRA Fine-Tuned**
- **Model:** [UDZH/whisper-small-lora-finetuned-ru](https://huggingface.co/UDZH/whisper-small-lora-finetuned-ru)
- **Usage Example:**
```python
from huggingface_hub import hf_hub_download
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper LoRA model
repo_id = "UDZH/whisper-small-lora-finetuned-ru"
lora_weights_path = hf_hub_download(repo_id=repo_id, filename="whisper_lora_weights.pth")

# Load base model and apply LoRA weights
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")

lora_weights = torch.load(lora_weights_path, map_location="cuda")
model.load_state_dict(lora_weights, strict=False)
```

### **DeepSeek Meeting Summary**
- **Model:** [UDZH/deepseek-meeting-summary](https://huggingface.co/UDZH/deepseek-meeting-summary)
- **Usage Example:**
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("UDZH/deepseek-meeting-summary")
tokenizer = AutoTokenizer.from_pretrained("UDZH/deepseek-meeting-summary")

# Example summarization
text = "This is a sample meeting transcript."
inputs = tokenizer(text, return_tensors="pt")
summary = model.generate(**inputs)
print(tokenizer.decode(summary[0], skip_special_tokens=True))
```

---

## 📓 Notebooks
### **1. `sample_implementation.ipynb`**
- **Purpose:** Runs the models on `meeting_video.mp4`.
- **Steps:**
  1. Extracts audio from the video
  2. Runs **Whisper LoRA** for speech recognition
  3. Runs **DeepSeek Summary** for meeting summarization
  4. Displays results

### **2. `sample_experiments.ipynb`**
- **Purpose:** Fine-tunes the models on **sample_datasets/**.
- **Steps:**
  1. Loads a subset of **Golos** and **Meeting** datasets
  2. Fine-tunes **Whisper LoRA** on Golos samples
  3. Fine-tunes **DeepSeek Summary** on meeting transcriptions
  4. Evaluates results

---

## 📈 Real Experiment Results
All real experiments and outputs are stored in `real_results/`.
- **`implementation.ipynb`** → Logs and transcriptions from running on full-sized examples
- **`experiments.ipynb`** → Logs from running on full-sized examples
- **`outputs/`** → CSV reports of model performance

---

## 🔧 Installation
### **1. Install dependencies**
```bash
pip install -r requirements.txt
```
Or you can install it from noteboooks directly.
---

## 📜 License
This repository is distributed under the **MIT License**.

---

## 👥 Contributors
- **@UDZH** – Model fine-tuning and dataset preparation
- **Uliana Dzhumok** – Repository setup and documentation (Optional)

---

## ⭐ Acknowledgements
Special thanks to:
- **Hugging Face** for hosting models
- **OpenAI** for Whisper
- **DeepSeek** for their models
- **Golos dataset creators**
- **AMI & ICSI Corpus providers**
- **Unsloth** for finetuning tools

