# Start from a Python base image that has everything we need for ML
FROM python:3.13-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the AI service code into the container
COPY app.py .
COPY models/ ./models/

# Expose the port the AI service will run on
EXPOSE 5002

# Define the command to start the AI service
CMD ["python", "app.py"]
