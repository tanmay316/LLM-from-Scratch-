
A GPT-style Small Language Model built from scratch using PyTorch, trained on the TinyStories dataset. This project demonstrates how to build, train, and deploy a compact language model capable of generating coherent text.

## 📊 Model Overview

- **Parameters**: ~30 Million
- **Architecture**: GPT-style Transformer
- **Dataset**: TinyStories (HuggingFace)
- **Model Size**: 114MB
- **Context Length**: 128 tokens
- **Vocabulary Size**: 50,257 tokens

## 🎯 Model Architecture

![Screenshot 2025-06-23 191655](https://github.com/user-attachments/assets/c235a5b8-c9bf-4813-be89-7eb6f988cbf9)
![Screenshot 2025-06-24 134741](https://github.com/user-attachments/assets/3685ab0a-c7c9-419f-bf87-72ed5d91cfbd)


```
Configuration:
├── Layers: 6
├── Attention Heads: 6  
├── Embedding Dimensions: 384
├── Dropout: 0.1
├── Block Size: 128
└── Bias: True
```


## 📚 Training Process

### Step 1: Dataset Preparation
**What it does**: Downloads and prepares the TinyStories dataset from HuggingFace.
**How it works**: TinyStories contains simple stories using vocabulary that 3-4 year olds understand, making it perfect for training small models efficiently.

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
```

### Step 2: Tokenization
**What it does**: Converts text into numerical tokens that the model can process.
**How it works**: Uses GPT-2's BPE tokenizer to split text into subword units, creating `train.bin` and `validation.bin` files for efficient data loading.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
# Converts text to token IDs and saves as memory-mapped files
```

### Step 3: Data Loading
**What it does**: Creates batches of input-output pairs for training.
**How it works**: Implements sliding window approach where each sequence predicts the next token. Uses memory mapping for efficient data access.

### Step 4: Model Architecture
**What it does**: Defines the GPT-style transformer architecture.
**How it works**: 
- **Embedding Layer**: Converts tokens to dense vectors
- **Positional Encoding**: Adds position information
- **Transformer Blocks**: 6 layers of self-attention + MLP
- **Layer Normalization**: Stabilizes training
- **Causal Attention**: Ensures autoregressive generation

### Step 5: Loss Function
**What it does**: Measures how well the model predicts next tokens.
**How it works**: Uses cross-entropy loss between predicted and actual next tokens, with evaluation on both training and validation sets.

### Step 6-7: Training Configuration
**What it does**: Sets up training hyperparameters and optimization.
**How it works**:
- **Learning Rate**: 1e-4 with cosine annealing
- **Warmup**: 1000 steps for stable initial training
- **Gradient Accumulation**: Simulates larger batch sizes
- **Mixed Precision**: fp16/bfloat16 for faster training

### Step 8: Training Loop
**What it does**: Trains the model for 20,000 iterations.
**How it works**:
- Forward pass through model
- Calculate loss and gradients
- Gradient clipping for stability
- Optimizer step with learning rate scheduling
- Regular evaluation and checkpointing

## 📈 Training Results

- **Training Loss**: Decreased from ~9.5 to ~2.3
- **Validation Loss**: Decreased from ~9.5 to ~2.3
- **Training Time**: ~20,000 iterations
- **Best Model**: Saved based on lowest validation loss



## 🔧 Training Configuration Details

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max Iterations | 20,000 | Total training steps |
| Learning Rate | 1e-4 | Initial learning rate |
| Batch Size | 32 | Samples per batch |
| Block Size | 128 | Context window |
| Gradient Accumulation | 32 | Effective batch size multiplier |
| Warmup Steps | 1,000 | Learning rate warmup |
| Min LR | 5e-4 | Final learning rate |
| Eval Interval | 500 | Evaluation frequency |


## 🛠️ Advanced Usage

### Custom Training
```python
# Modify config for different model sizes
config.n_layer = 8      # More layers
config.n_embd = 512     # Larger embeddings
config.block_size = 256 # Longer context
```

## 🔬 Technical Details

### Architecture Choices
- **Causal Attention**: Ensures autoregressive generation
- **Weight Tying**: Shares embedding and output weights
- **Layer Normalization**: Pre-norm configuration for stability
- **GELU Activation**: Smooth activation function
- **Dropout**: 0.1 for regularization

### Training Optimizations
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Faster training with minimal quality loss
- **Learning Rate Scheduling**: Warmup + cosine annealing
- **Gradient Accumulation**: Simulates larger batch sizes


## 📞 Contact

For questions or suggestions, please open an issue or reach out through GitHub.

---
