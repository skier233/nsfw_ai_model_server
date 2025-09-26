#!/bin/bash

# Activate conda environment and start server
conda deactivate
conda remove -n ai_model_server --all
if [ "$1" = '--amd' ]; then
    echo "AMD-Mode activated!"
    # Create Env with special libs
    conda env create -f environment-linux-amd.yml
    source ./install-amd-post.sh
    # Build AMD Docker image (uses base Dockerfile)
    echo "Building AMD Docker image..."
    docker build -t ai-model-server-amd .
elif [ "$1" = '--intel' ]; then
    echo "Intel-Mode activated!"
    # Create Env with Intel-specific libs
    # Build Intel Docker image (uses Intel-specific Dockerfile)
    echo "Building Intel Docker image..."
    docker build -f Dockerfile-intel -t ai-model-server-intel .
    echo "~~ Make sure to pass --device /dev/dri to pass your Intel GPU into your container~~"
else
    conda env create -f environment-linux.yml
    conda activate ai_model_server
    # Build default Docker image
    echo "Building default Docker image..."
    docker build -t ai-model-server .
fi
echo "Docker image built as ai-model-server"
python migrate.py
python server.py