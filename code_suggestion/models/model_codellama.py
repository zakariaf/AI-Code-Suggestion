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

    def generate(self, context: str, max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2) -> str:
        """
        Generates text based on the given context.

        Args:
            context (str): The input context string.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 200.
            temperature (float, optional): The temperature for generation. Defaults to 0.7.
            top_k (int, optional): The top-k value for sampling. Defaults to 50.
            top_p (float, optional): The top-p (nucleus) value for sampling. Defaults to 0.95.
            repetition_penalty (float, optional): The repetition penalty for generation. Defaults to 1.2.

        Returns:
            str: The generated text.  Returns an error message if there's an issue.
        """
        try:
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,  # Explicitly set do_sample
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,  # Essential for correct padding
                eos_token_id=self.tokenizer.eos_token_id # Stop generation when EOS token is encountered
            )

            # Decode the result
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            return f"Error during generation: {e}"
