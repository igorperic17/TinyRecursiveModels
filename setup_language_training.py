#!/usr/bin/env python3
"""
Setup script for language generation training with TRM.
This creates a minimal dataset and sets up everything for quick iteration.
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*50)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        return False
    else:
        print(f"Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Setup language training for TRM")
    parser.add_argument("--task", choices=["conversation", "story", "qa", "code"], 
                       default="conversation", help="Task type for the dataset")
    parser.add_argument("--num-examples", type=int, default=1000, 
                       help="Number of examples to generate")
    parser.add_argument("--max-length", type=int, default=128, 
                       help="Maximum sequence length")
    parser.add_argument("--tokenizer", default="microsoft/DialoGPT-small", 
                       help="Tokenizer to use")
    
    args = parser.parse_args()
    
    print("Setting up language training for TRM...")
    print(f"Task: {args.task}")
    print(f"Examples: {args.num_examples}")
    print(f"Max length: {args.max_length}")
    print(f"Tokenizer: {args.tokenizer}")
    
    # Create data directory
    data_dir = "data/language-minimal"
    os.makedirs(data_dir, exist_ok=True)
    
    # Build the dataset
    cmd = f"""python dataset/build_language_dataset.py \\
        --output-dir {data_dir} \\
        --tokenizer-name {args.tokenizer} \\
        --max-length {args.max_length} \\
        --num-examples {args.num_examples} \\
        --task-type {args.task}"""
    
    if not run_command(cmd, "Building language dataset"):
        print("Failed to build dataset")
        return False
    
    print(f"\nDataset created at: {data_dir}")
    print("\nNext steps:")
    print("1. Run training: python train_language.py")
    print("2. Or use the full training pipeline: python pretrain.py arch=trm_language data_paths=[data/language-minimal]")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
