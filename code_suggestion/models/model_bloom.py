"""
model_bloom.py - Code specific to loading and generating text with Bloom.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BloomModel:
    def __init__(self, model_name="bigscience/bloom-560m"):
        """
        Initialize the tokenizer and model for Bloom.
        Default: bigscience/bloom-560m, a small Bloom model for demonstration.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, context: str) -> str:
        """
        Generate text given a context string.
        """
        # Tokenize
        inputs = self.tokenizer(context, return_tensors="pt")

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + 30,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Decode
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion
