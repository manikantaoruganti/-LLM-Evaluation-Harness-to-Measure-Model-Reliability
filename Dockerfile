# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies if any (none for this project)

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create results directory if it doesn't exist (for volume mounting)
RUN mkdir -p results

# Default command to run the evaluation
CMD ["python", "-m", "scripts.run_evaluation"]
