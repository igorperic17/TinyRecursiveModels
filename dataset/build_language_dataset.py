from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer

from argdantic import ArgParser
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str
    tokenizer_name: str = "microsoft/DialoGPT-small"  # Small tokenizer for quick iteration
    max_length: int = 128
    seed: int = 42
    num_examples: int = 1000  # Small dataset for quick iteration
    task_type: str = "conversation"  # conversation, story, qa, code


@dataclass
class LanguageExample:
    id: str
    input_text: str
    output_text: str
    task_type: str


def create_minimal_dataset(config: DataProcessConfig) -> List[LanguageExample]:
    """Create a minimal dataset for quick iteration."""
    
    examples = []
    
    if config.task_type == "conversation":
        # Simple conversation examples
        conversations = [
            ("Hello, how are you?", "I'm doing well, thank you for asking! How about you?"),
            ("What's the weather like?", "I don't have access to real-time weather data, but I'd be happy to help you find weather information online."),
            ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
            ("What's 2+2?", "2+2 equals 4."),
            ("Can you help me with coding?", "Of course! I'd be happy to help you with coding. What programming language or problem are you working on?"),
            ("Explain quantum computing", "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot."),
            ("What's your favorite color?", "I don't have personal preferences, but I find the concept of color fascinating from a scientific perspective!"),
            ("How do I learn programming?", "Start with a beginner-friendly language like Python, practice regularly, build projects, and don't be afraid to make mistakes - they're part of learning!"),
        ]
        
        # Duplicate and vary for more examples
        for i in range(config.num_examples // len(conversations) + 1):
            for j, (input_text, output_text) in enumerate(conversations):
                if len(examples) >= config.num_examples:
                    break
                examples.append(LanguageExample(
                    id=f"conv_{i}_{j}",
                    input_text=input_text,
                    output_text=output_text,
                    task_type="conversation"
                ))
    
    elif config.task_type == "story":
        # Simple story completion
        story_prompts = [
            "Once upon a time, there was a brave knight who",
            "In a small village, a mysterious door appeared",
            "The last robot on Earth discovered",
            "A young wizard's first spell went wrong when",
            "Deep in the forest, an ancient tree began to",
        ]
        
        story_completions = [
            "ventured into the dark forest to rescue the captured princess from the dragon's lair.",
            "in the town square. No one knew where it came from, but it seemed to call to those who passed by.",
            "that it wasn't alone - other robots had been hiding in underground bunkers for years.",
            "he tried to turn his homework into chocolate cake, but instead turned his teacher into a giant hamster.",
            "speak, revealing secrets that had been hidden for centuries about the magical realm.",
        ]
        
        for i in range(config.num_examples):
            prompt_idx = i % len(story_prompts)
            completion_idx = i % len(story_completions)
            examples.append(LanguageExample(
                id=f"story_{i}",
                input_text=story_prompts[prompt_idx],
                output_text=story_prompts[prompt_idx] + " " + story_completions[completion_idx],
                task_type="story"
            ))
    
    elif config.task_type == "qa":
        # Simple Q&A pairs
        qa_pairs = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy."),
            ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
            ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
        ]
        
        for i in range(config.num_examples):
            qa_idx = i % len(qa_pairs)
            examples.append(LanguageExample(
                id=f"qa_{i}",
                input_text=qa_pairs[qa_idx][0],
                output_text=qa_pairs[qa_idx][1],
                task_type="qa"
            ))
    
    elif config.task_type == "code":
        # Simple code generation
        code_prompts = [
            ("Write a function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
            ("Create a function to reverse a string", "def reverse_string(s):\n    return s[::-1]"),
            ("Write a function to check if a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
            ("Create a function to find the maximum in a list", "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"),
        ]
        
        for i in range(config.num_examples):
            code_idx = i % len(code_prompts)
            examples.append(LanguageExample(
                id=f"code_{i}",
                input_text=code_prompts[code_idx][0],
                output_text=code_prompts[code_idx][1],
                task_type="code"
            ))
    
    return examples[:config.num_examples]


def tokenize_examples(examples: List[LanguageExample], tokenizer, max_length: int) -> Dict[str, List]:
    """Tokenize examples using the provided tokenizer."""
    
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [],
        "group_indices": []
    }
    
    puzzle_indices = [0]
    group_indices = [0]
    example_id = 0
    puzzle_id = 0
    
    for example in examples:
        # Tokenize input and output
        input_tokens = tokenizer.encode(example.input_text, add_special_tokens=True, max_length=max_length, truncation=True)
        output_tokens = tokenizer.encode(example.output_text, add_special_tokens=True, max_length=max_length, truncation=True)
        
        # Create full sequence (input + output)
        full_sequence = input_tokens + output_tokens[1:]  # Remove duplicate BOS token
        
        # Pad to max_length
        if len(full_sequence) > max_length:
            full_sequence = full_sequence[:max_length]
        
        # Create input and label sequences
        input_seq = full_sequence[:-1]  # All tokens except last
        label_seq = full_sequence[1:]   # All tokens except first
        
        # Pad sequences
        input_seq = input_seq + [tokenizer.pad_token_id] * (max_length - len(input_seq))
        label_seq = label_seq + [-100] * (max_length - len(label_seq))  # -100 for ignore index
        
        results["inputs"].append(input_seq)
        results["labels"].append(label_seq)
        results["puzzle_identifiers"].append(0)  # No puzzle-specific embeddings for language
        
        example_id += 1
        puzzle_indices.append(example_id)
        
        puzzle_id += 1
        group_indices.append(puzzle_id)
    
    results["puzzle_indices"] = puzzle_indices
    results["group_indices"] = group_indices
    
    return results


def convert_dataset(config: DataProcessConfig):
    """Convert the dataset to the required format."""
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create examples
    print(f"Creating {config.num_examples} examples for task: {config.task_type}")
    examples = create_minimal_dataset(config)
    
    # Tokenize
    print("Tokenizing examples...")
    results = tokenize_examples(examples, tokenizer, config.max_length)
    
    # Convert to numpy arrays
    for k, v in results.items():
        if k in {"inputs", "labels"}:
            results[k] = np.array(v, dtype=np.int32)
        else:
            results[k] = np.array(v, dtype=np.int32)
    
    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_length,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
        ignore_label_id=-100,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1.0,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )
    
    # Save data
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(config.output_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    # Save data arrays
    for k, v in results.items():
        np.save(os.path.join(config.output_dir, f"all__{k}.npy"), v)
    
    print(f"Dataset saved to {config.output_dir}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sequence length: {config.max_length}")
    print(f"Total examples: {len(examples)}")


@cli.command()
def main(
    output_dir: str,
    tokenizer_name: str = "microsoft/DialoGPT-small",
    max_length: int = 128,
    seed: int = 42,
    num_examples: int = 1000,
    task_type: str = "conversation"
):
    """Build a minimal language dataset for quick iteration."""
    
    config = DataProcessConfig(
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        seed=seed,
        num_examples=num_examples,
        task_type=task_type
    )
    
    convert_dataset(config)


if __name__ == "__main__":
    cli()
