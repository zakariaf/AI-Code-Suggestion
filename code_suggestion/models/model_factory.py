"""
model_factory.py - Chooses which model class to instantiate based on environment variables.
"""

import os
from models.model_bloom import BloomModel
from models.model_codellama import CodeLlamaModel

def create_model():
    """
    Reads the MODEL_TYPE environment variable and instantiates the appropriate model class.
    """
    # Read environment variable MODEL_TYPE, defaulting to 'bloom'
    model_type = os.getenv("MODEL_TYPE", "bloom").lower()

    # Decide which class to instantiate
    if model_type == "bloom":
        return BloomModel()
    elif model_type == "codellama":
        return CodeLlamaModel()
    else:
        # If no matching type found, default to Bloom
        return BloomModel()
