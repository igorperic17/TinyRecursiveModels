#!/usr/bin/env python3
"""
Quick training script for language generation with TRM.
This is a simplified version focused on quick iteration.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.recursive_reasoning.trm_language import TinyRecursiveLanguageModel_ACTV1
from models.losses import ACTLossHead, stablemax_cross_entropy
from dataset.build_language_dataset import create_minimal_dataset, tokenize_examples


class SimpleLanguageDataset:
    """Simple dataset for language generation."""
    
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        input_tokens = self.tokenizer.encode(example.input_text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        output_tokens = self.tokenizer.encode(example.output_text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        
        # Create full sequence
        full_sequence = input_tokens + output_tokens[1:]  # Remove duplicate BOS
        
        if len(full_sequence) > self.max_length:
            full_sequence = full_sequence[:self.max_length]
        
        # Create input and label sequences
        input_seq = full_sequence[:-1]
        label_seq = full_sequence[1:]
        
        # Pad
        input_seq = input_seq + [self.tokenizer.pad_token_id] * (self.max_length - len(input_seq))
        label_seq = label_seq + [-100] * (self.max_length - len(label_seq))
        
        return {
            "inputs": torch.tensor(input_seq, dtype=torch.long),
            "labels": torch.tensor(label_seq, dtype=torch.long)
        }


def train_step(model, batch, optimizer, device):
    """Single training step with error handling."""
    try:
        model.train()
        
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Initialize carry
        carry = model.initial_carry(batch)
        
        # Forward pass
        carry, outputs = model(carry, batch)
        
        # Compute loss
        logits = outputs["logits"]
        labels = batch["labels"]
        
        # Standard language modeling loss
        loss = stablemax_cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.mean()
        
        # ACT loss (if enabled)
        if "q_halt_logits" in outputs:
            halt_logits = outputs["q_halt_logits"]
            # Simple ACT loss: encourage halting at appropriate times
            act_loss = torch.sigmoid(halt_logits).mean()
            loss = loss + 0.1 * act_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    except Exception as e:
        print(f"Error in training step: {e}")
        return float('inf')


def generate_text(model, tokenizer, prompt, max_length=50, device="cuda"):
    """Generate text using the model."""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Initialize carry
    batch = {"inputs": input_ids}
    carry = model.initial_carry(batch)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            carry, outputs = model(carry, batch)
            
            # Get next token
            logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            
            # Check if we should halt
            if "q_halt_logits" in outputs:
                halt_prob = torch.sigmoid(outputs["q_halt_logits"])
                if halt_prob.mean() > 0.5:  # Simple halting criterion
                    break
            
            # Add to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            
            # Update batch for next iteration
            batch = {"inputs": generated}
    
    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


@hydra.main(config_path="config", config_name="cfg_language_pretrain", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_name = "microsoft/DialoGPT-small"
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating dataset...")
    examples = create_minimal_dataset(type('Config', (), {
        'num_examples': 200,  # Smaller dataset for quick training
        'task_type': 'conversation'
    })())
    
    # Split into train/val
    train_examples = examples[:160]
    val_examples = examples[160:]
    
    # Create datasets
    train_dataset = SimpleLanguageDataset(train_examples, tokenizer, max_length=64)  # Shorter sequences
    val_dataset = SimpleLanguageDataset(val_examples, tokenizer, max_length=64)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Smaller batch
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model (smaller for quick training)
    model_config = {
        "batch_size": 4,
        "seq_len": 64,
        "vocab_size": tokenizer.vocab_size,
        "H_cycles": 1,  # Fewer cycles
        "L_cycles": 2,
        "H_layers": 0,  # Not used but required
        "L_layers": 1,  # Fewer layers
        "hidden_size": 128,  # Smaller model
        "num_heads": 4,
        "expansion": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "causal": True,
        "no_ACT_continue": True
    }
    
    model = TinyRecursiveLanguageModel_ACTV1(model_config)
    model = model.to(device)
    
    # Note: Using model directly for now, not wrapped in ACTLossHead
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(5):  # Fewer epochs for quick training
        total_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/5")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 20:  # Limit batches to prevent freezing
                break
                
            loss = train_step(model, batch, optimizer, device)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx + 1}: Loss = {loss:.4f}")
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1} failed - no successful batches")
        
        # Generate some text
        if epoch % 2 == 0:
            print("\nGenerating text...")
            test_prompts = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Tell me a story about",
            ]
            
            for prompt in test_prompts:
                generated = generate_text(model, tokenizer, prompt, max_length=30, device=device)
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated}")
                print()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
