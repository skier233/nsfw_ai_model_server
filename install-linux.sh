#!/bin/bash

# Activate conda environment and start server
conda deactivate
conda remove -n ai_model_server --all
if [ "$1" = '--amd' ]; then
    echo "AMD-Mode activated!"
    # Create Env with special libs
    conda env create -f environment-linux-amd.yml
    source ./install-amd-post.sh
else
    conda env create -f environment-linux.yml
    conda activate ai_model_server
fi
python migrate.py
python server.py