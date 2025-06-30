# Fine-tune BioMedLM with MedQuAD Dataset

A comprehensive implementation for fine-tuning the BioMedLM (2.7B parameter) model using Low-Rank Adaptation (LoRA) on the MedQuAD dataset containing ~47,000 medical question-answer pairs.

## 🎯 Project Overview

This project enhances BioMedLM's ability to provide accurate, evidence-based answers to medical questions by leveraging its biomedical pre-training and applying efficient fine-tuning techniques. The implementation uses LoRA (Low-Rank Adaptation) for parameter-efficient training, making it feasible to run on consumer GPUs.

## 🚀 Key Features

- **Memory Efficient**: Uses 4-bit quantization and LoRA for ~75% memory reduction
- **High Performance**: Maintains model quality while training only ~1% of parameters
- **Production Ready**: Saves models in SafeTensors format for portability
- **Comprehensive Evaluation**: Includes ROUGE scores and sample testing
- **Well Documented**: Detailed explanations and configuration options

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3090, A100)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~50GB free space for models and datasets

### Software Requirements
- Python 3.8+
- CUDA 11.8+ compatible GPU drivers
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/naveeeen-ai/Fine-tune-BioMedLM.git
cd Fine-tune-BioMedLM
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv biomedlm_env
source biomedlm_env/bin/activate  # On Windows: biomedlm_env\Scripts\activate
```

### 3. Install Dependencies
Run the first cell in the `Finetune.ipynb` notebook, or install manually:
```bash
pip install transformers==4.36.0
pip install peft==0.7.1
pip install bitsandbytes==0.41.3
pip install accelerate==0.25.0
pip install datasets==2.16.0
pip install safetensors==0.4.1
pip install rouge-score==0.1.2
pip install nltk==3.8.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Dataset

The project uses the **MedQuAD dataset** which contains:
- **Size**: ~47,000 medical question-answer pairs
- **Sources**: Multiple medical institutions and databases
- **Format**: CSV with columns: `question`, `answer`, `source`, `focus_area`
- **Split**: 80% training, 20% validation

### Data Format Example
```
Question: What are the main symptoms of diabetes?
Answer: The main symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, cuts and bruises that are slow to heal, and weight loss despite increased appetite...
```

## 🤖 Model Details

### Base Model: BioMedLM
- **Parameters**: 2.7 billion
- **Architecture**: Transformer-based language model
- **Pre-training**: Biomedical literature and clinical texts
- **Publisher**: Stanford CRFM

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32 (2x scaling factor)
- **Target Modules**: All attention and MLP projection layers
- **Dropout**: 0.1
- **Trainable Parameters**: ~0.1% of total parameters

## ⚙️ Training Configuration

### Optimizations Used
- **4-bit Quantization (NF4)**: Reduces memory usage by 75%
- **Gradient Checkpointing**: Trades compute for memory
- **Mixed Precision (FP16)**: 50% memory reduction
- **Gradient Accumulation**: Effective batch size of 16
- **Paged AdamW 8-bit**: Memory-efficient optimizer

### Hyperparameters
- **Learning Rate**: 2e-5
- **Batch Size**: 4 per device (effective: 16 with accumulation)
- **Epochs**: 3
- **Max Sequence Length**: 512 tokens
- **Warmup Steps**: 100
- **Weight Decay**: 0.01

## 🚀 Usage

### Quick Start
1. Open `Finetune.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially from top to bottom
3. Monitor training progress and metrics
4. Evaluate the fine-tuned model

### Running Training
```python
# The notebook handles everything, but key steps are:
# 1. Load and preprocess data
# 2. Configure model with LoRA
# 3. Set up training arguments
# 4. Train the model
# 5. Evaluate and save results
```

### Using the Fine-tuned Model
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("stanford-crfm/BioMedLM")
tokenizer = AutoTokenizer.from_pretrained("./biomedlm-lora-medquad/safetensors")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./biomedlm-lora-medquad/safetensors")

# Generate answer
question = "What are the symptoms of COVID-19?"
answer = generate_medical_answer(question, model, tokenizer)
print(answer)
```

## 📈 Expected Results

### Performance Metrics
Based on similar fine-tuning tasks:
- **ROUGE-1**: 0.35-0.45
- **ROUGE-2**: 0.15-0.25
- **ROUGE-L**: 0.30-0.40
- **Exact Match**: 0.05-0.15

### Training Time
- **RTX 3090**: ~3-6 hours
- **A100**: ~2-4 hours
- **Memory Usage**: ~12-14GB VRAM with optimizations

## 📁 File Structure

```
Fine-tune-BioMedLM/
├── README.md                          # This file
├── Finetune.ipynb                     # Main training notebook
├── medquad.csv                        # MedQuAD dataset
└── biomedlm-lora-medquad/             # Generated during training
    ├── safetensors/                   # LoRA adapter (~50-100MB)
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   └── tokenizer files
    ├── merged_model/                  # Full merged model (~10GB)
    ├── logs/                          # TensorBoard logs
    └── evaluation_results.json        # Evaluation metrics
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch size or enable more aggressive quantization
- Set BATCH_SIZE = 2 or 1
- Enable LOAD_IN_8BIT = True
- Reduce MAX_LENGTH to 256
```

**2. Installation Issues**
```
Solution: Ensure compatible versions
- Use Python 3.8-3.11
- Install PyTorch with correct CUDA version
- Use virtual environment to avoid conflicts
```

**3. Slow Training**
```
Solution: Optimize settings
- Enable gradient checkpointing
- Use tensor cores (batch size multiple of 8)
- Monitor GPU utilization
```

**4. Poor Generation Quality**
```
Solution: Adjust generation parameters
- Lower temperature (0.3-0.5) for more deterministic output
- Adjust top_p (0.8-0.95)
- Increase max_new_tokens for longer responses
```

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./biomedlm-lora-medquad/logs
```

### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without diverging from training loss
- **Learning Rate**: Should follow warmup schedule
- **GPU Utilization**: Should be >80% during training

## 🔬 Evaluation

The notebook includes comprehensive evaluation:
- **Automatic Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, Exact Match
- **Sample Predictions**: Manual inspection of model outputs
- **Medical Question Testing**: Predefined medical questions

## 🚨 Important Notes

### Medical Disclaimer
- This model is for research purposes only
- Not intended for actual medical diagnosis or treatment
- Always consult qualified healthcare professionals
- Implement appropriate safety checks for production use

### Ethical Considerations
- Ensure responsible use of medical AI
- Implement bias detection and mitigation
- Consider fairness across different demographics
- Maintain data privacy and security

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stanford CRFM** for the BioMedLM model
- **MedQuAD Dataset** creators for the comprehensive medical Q&A dataset
- **Hugging Face** for the transformers library and PEFT implementation
- **Microsoft** for the LoRA technique

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the notebook comments for detailed explanations

## 🔗 Related Resources

- [BioMedLM Paper](https://arxiv.org/abs/2403.18421)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MedQuAD Dataset](https://github.com/abachaa/MedQuAD)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)

---

**Happy Fine-tuning! 🚀** 