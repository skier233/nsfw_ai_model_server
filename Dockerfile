# Use an official Python runtime as a parent image with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.11 python3.11-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Set Python 3.11 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip dependencies
COPY requirements.txt .
RUN python3.11 -m pip install -r requirements.txt

# Copy the wheel file and install it
COPY dist/ai_processing-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/
RUN python3.11 -m pip install /tmp/ai_processing-0.0.0-cp311-cp311-linux_x86_64.whl

# Expose the port FastAPI runs on
EXPOSE 8000

# Set the working directory
WORKDIR /app

# Command to run the server.py script
CMD ["python3.11", "server.py"]
