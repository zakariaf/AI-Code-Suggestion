"""
app.py - A simple Flask app acting as an AI microservice.
Loads a model from Hugging Face and returns code suggestions.
"""

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)  # Initialize the Flask app

# Load or initialize your model/tokenizer (this could be Code Llama or GPT-like model).
# For demonstration, let's assume "huggingface/CodeLlama" is a placeholder.
MODEL_NAME = "bigscience/bloom-560m"  # Example: smaller model to keep it simpler
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Route for generating code suggestions
@app.route("/generate_suggestion", methods=["POST"])
def generate_suggestion():
    """
    This endpoint receives a JSON payload with the partial code or context,
    and returns a code suggestion from the model.
    """
    # Extract the JSON data from the request
    data = request.get_json()

    # 'context' might be the partial code snippet the user typed
    context = data.get("context", "")

    # Tokenize the context to prepare for model inference
    inputs = tokenizer(context, return_tensors="pt")

    # Generate output from the model
    #    'max_length' and other params are adjustable
    outputs = model.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0]) + 30,  # generate up to 30 tokens more
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode the output tokens to get text
    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the suggestion in JSON format
    return jsonify({"suggestion": suggestion})

# Start the Flask app if running directly
if __name__ == "__main__":
    # Host set to 0.0.0.0 to allow external Docker traffic, port 5002
    app.run(host="0.0.0.0", port=5002)
