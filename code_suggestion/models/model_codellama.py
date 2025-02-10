"""
model_codellama.py - Code specific to loading and generating text with Code Llama.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CodeLlamaModel:
    def __init__(self, model_name="codellama/CodeLlama-7b-Instruct-hf", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the tokenizer and model for Code Llama.

        Args:
            model_name (str, optional): The name of the CodeLlama model to use. Defaults to "codellama/CodeLlama-7b-Instruct-hf".
            device (str, optional): The device to use for the model. Defaults to "cuda" if a GPU is available, otherwise "cpu".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, context: str) -> str:
        """
        Generate text given a context string.
        """
        # Tokenize context
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

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
