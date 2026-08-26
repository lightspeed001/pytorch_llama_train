# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*
        
# Create and activate Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face transformers, datasets, and other dependencies
RUN pip install --no-cache-dir \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    datasets==2.18.0 \
    peft==0.8.2 \
    bitsandbytes==0.43.0 \
    sentencepiece==0.1.99 \
    scikit-learn==1.4.0 \
    ninja==1.11.1

# Copy the training script into the container
COPY llama_train.py /app/llama_train.py

# Set working directory
WORKDIR /app

# Default command (override in k8s config)
CMD ["python", "train.py"]
