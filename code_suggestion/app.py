"""
app.py - A simple Flask app acting as an AI microservice.
loads a chosen model and returns code suggestions.
"""

from flask import Flask, request, jsonify
from models.model_factory import create_model  # Import the factory method

app = Flask(__name__)  # Initialize the Flask app

# (2) Create the model instance once, at startup
#     This will load whichever model is selected by environment variable
model_instance = create_model()

@app.route("/generate_suggestion", methods=["POST"])
def generate_suggestion():
    """
    This endpoint receives a JSON payload with 'context',
    and returns a code suggestion from the chosen model.
    """
    # Extract the JSON data from the request
    data = request.get_json()

    # 'context' might be the partial code snippet the user typed
    context = data.get("context", "")

    # Use the chosen model to generate code
    suggestion = model_instance.generate(context)

    # Return the suggestion in JSON format
    return jsonify({"suggestion": suggestion})

# Start the Flask app if running directly
if __name__ == "__main__":
    # Run on 0.0.0.0:5002 so Docker can expose it
    app.run(host="0.0.0.0", port=5002)
