"""
model_codellama.py - Code specific to loading and generating text with Code Llama.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CodeLlamaModel:
    def __init__(self, model_name="codellama/CodeLlama-7b-Instruct-hf"):
        """
        Initialize the tokenizer and model for Code Llama.
        Example: 'codellama/CodeLlama-7b-Instruct-hf'
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, context: str) -> str:
        """
        Generate text given a context string.
        """
        # Tokenize context
        inputs = self.tokenizer(context, return_tensors="pt")

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + 30,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Decode the result
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion
