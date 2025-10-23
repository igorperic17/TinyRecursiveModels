# Language Generation with Tiny Recursive Models

This guide shows how to adapt the TRM architecture for language generation with standard LLM tokenization and quick iteration.

## Quick Start

### 1. Setup Environment

**For macOS (Apple Silicon):**
```bash
# Use the macOS-specific setup script
./setup_macos.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
```

**For Linux/Windows:**
```bash
# Install dependencies (if not already done)
pip install -r requirements.txt
pip install transformers  # For tokenization
```

### 2. Create Minimal Dataset
```bash
# Create a small conversation dataset (1000 examples)
python setup_language_training.py --task conversation --num-examples 1000

# Or create other types:
python setup_language_training.py --task story --num-examples 500
python setup_language_training.py --task qa --num-examples 800
python setup_language_training.py --task code --num-examples 600
```

### 3. Train the Model

**Option A: Quick training script (recommended for quick iteration)**
```bash
python train_language.py
```

**Option B: Full training pipeline**
```bash
python pretrain.py arch=trm_language data_paths=[data/language-minimal]
```

## Architecture Changes for Language Generation

### Key Modifications Made:

1. **Removed puzzle-specific embeddings**: No more `puzzle_identifiers` or `puzzle_emb`
2. **Standard tokenization**: Uses HuggingFace tokenizers (DialoGPT-small by default)
3. **Causal attention**: Added `causal=True` for autoregressive generation
4. **Simplified input processing**: Direct token embedding without puzzle-specific components
5. **Shorter reasoning cycles**: Reduced `H_cycles` and `L_cycles` for language tasks

### Model Configuration:

```yaml
# config/arch/trm_language.yaml
name: recursive_reasoning.trm_language@TinyRecursiveLanguageModel_ACTV1
halt_max_steps: 8  # Shorter for language
H_cycles: 2       # Fewer cycles
L_cycles: 3
hidden_size: 256   # Smaller for quick iteration
num_heads: 4
causal: True       # Causal attention
```

## Dataset Format

The language dataset uses standard LLM format:
- **Input**: Tokenized prompt text
- **Output**: Tokenized response text  
- **Format**: Standard language modeling (input tokens + output tokens)
- **Padding**: Uses tokenizer's pad token
- **Special tokens**: BOS/EOS tokens handled by tokenizer

## Training Features

### Quick Iteration Setup:
- **Small datasets**: 1000 examples for fast training
- **Short sequences**: 128 tokens max length
- **Small model**: 256 hidden size, ~2M parameters
- **Fast training**: 10 epochs, quick convergence

### Recursive Reasoning for Language:
1. **Input encoding**: Token embeddings + positional encoding
2. **Recursive updates**: 
   - Update latent `z_L` given input and current `z_H`
   - Update latent `z_H` given current `z_L`
3. **Progressive improvement**: Each reasoning step can refine the response
4. **Adaptive halting**: Model learns when to stop reasoning

## Example Usage

### Generate Text:
```python
from models.recursive_reasoning.trm_language import TinyRecursiveLanguageModel_ACTV1
from transformers import AutoTokenizer

# Load model and tokenizer
model = TinyRecursiveLanguageModel_ACTV1(config)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# Generate text
prompt = "Hello, how are you?"
generated = generate_text(model, tokenizer, prompt, max_length=50)
print(generated)
```

### Training Loop:
```python
# Standard training with recursive reasoning
for batch in dataloader:
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)
    
    # Compute loss
    loss = stablemax_cross_entropy(outputs["logits"], batch["labels"])
    loss.backward()
    optimizer.step()
```

## Scaling Up

Once you have a working model, you can scale up:

1. **Larger datasets**: Increase `--num-examples` to 10K, 100K, etc.
2. **Longer sequences**: Increase `--max-length` to 256, 512, etc.
3. **Bigger models**: Increase `hidden_size` to 512, 1024, etc.
4. **More reasoning**: Increase `H_cycles` and `L_cycles`
5. **Better tokenizers**: Use larger tokenizers like GPT-2, LLaMA, etc.

## Task Types

### Conversation
- Simple Q&A pairs
- Casual conversation examples
- Good for testing basic language understanding

### Story
- Story completion tasks
- Creative writing prompts
- Tests narrative reasoning

### Q&A
- Factual question answering
- Knowledge-based reasoning
- Tests factual recall and reasoning

### Code
- Code generation tasks
- Programming problem solving
- Tests logical reasoning and syntax

## macOS-Specific Features

### MLX Integration:
- **MLX Framework**: Optimized for Apple Silicon (M1/M2/M3)
- **MPS Backend**: Uses Metal Performance Shaders for GPU acceleration
- **No CUDA Required**: Works natively on macOS without CUDA dependencies

### Performance Benefits:
- **Apple Silicon Optimization**: Leverages M-series chip capabilities
- **Memory Efficiency**: Better memory management on Apple hardware
- **Native Performance**: No emulation overhead

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Slow training**: Use smaller model or fewer examples
3. **Poor generation**: Increase training epochs or model size
4. **Tokenization errors**: Check tokenizer compatibility

### macOS-Specific Issues:

1. **MLX import errors**: Ensure you're using native ARM64 Python
2. **MPS not available**: Check macOS version (13.5+ required)
3. **adam-atan2 errors**: Use the included macOS-compatible version
4. **Performance issues**: Ensure you're on Apple Silicon for best performance

### Performance Tips:

1. **Start small**: Use minimal datasets for initial testing
2. **Monitor loss**: Watch for convergence patterns
3. **Test generation**: Regularly check generated text quality
4. **Scale gradually**: Increase complexity step by step

## Next Steps

1. **Experiment with different tasks**: Try story, Q&A, or code generation
2. **Tune hyperparameters**: Adjust learning rates, model size, reasoning cycles
3. **Scale up datasets**: Use larger, more diverse training data
4. **Compare with baselines**: Test against standard transformers
5. **Analyze reasoning**: Study how the recursive reasoning improves generation quality
