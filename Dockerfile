# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., for Qt/GUI if needed, though mostly headless here)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Install the package
RUN pip install --no-cache-dir -e .

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run verification smoke test by default
CMD ["eqprop-verify", "--quick"]
