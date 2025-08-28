#!/bin/bash

# Activate conda environment and start server
conda deactivate
conda remove -n ai_model_server --all
if [ "$1" = '--amd' ]; then
    echo "AMD-Mode activated!"
    # Create Env with special libs
    conda env create -f environment-linux-amd.yml
    conda activate ai_model_server
    # Follow https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html
    location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
    rm ${location}/torch/lib/libhsa-runtime64.so*
    conda install -c conda-forge gcc=12.1.0
else
    conda env create -f environment-linux.yml
    conda activate ai_model_server
fi
python migrate.py
python server.py