# Fine-tune BioMedLM with MedQuAD Dataset

Fine-tune the BioMedLM (2.7B) model using LoRA on ~47,000 medical question-answer pairs.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/naveeeen-ai/Fine-tune-BioMedLM.git
   cd Fine-tune-BioMedLM
   ```

2. **Create virtual environment (IMPORTANT!)**
   ```bash
   python -m venv biomedlm_env
   source biomedlm_env/bin/activate  # On Windows: biomedlm_env\Scripts\activate
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Finetune.ipynb
   ```

4. **Run all cells** - The notebook handles everything automatically!

## 📋 Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090 or better)
- **Python**: 3.8+
- **CUDA**: 11.8+

## 📊 What's Included

- `Finetune.ipynb` - Complete training notebook
- `medquad.csv` - Medical Q&A dataset (~47k pairs)
- Automatic dependency installation
- LoRA fine-tuning with memory optimization
- Model evaluation and saving

## ⚙️ Key Features

- **Memory Efficient**: Uses 4-bit quantization + LoRA
- **Fast Training**: ~3-6 hours on RTX 3090
- **Easy to Use**: Just run the notebook cells
- **Production Ready**: Saves in SafeTensors format

## 🔧 Troubleshooting

**Out of Memory?**
- Reduce `BATCH_SIZE` to 2 or 1
- Enable `LOAD_IN_8BIT = True`

**Installation Issues?**
- Use Python 3.8-3.11
- Create a virtual environment

## 📝 Usage After Training

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")
model = PeftModel.from_pretrained(base_model, "./biomedlm-lora-medquad/safetensors")
tokenizer = AutoTokenizer.from_pretrained("./biomedlm-lora-medquad/safetensors")

# Ask medical questions
question = "What are the symptoms of diabetes?"
# Use the generate_medical_answer function from the notebook
```

## ⚠️ Important Note

This model is for research purposes only. Not for medical diagnosis. Always consult healthcare professionals.

---

**Happy Fine-tuning! 🚀** 