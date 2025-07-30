#!/bin/bash

# Activate conda environment and start server
export CONDA_SUBDIR=osx-arm64
conda deactivate
conda remove -n ai_model_server --all
conda env create -f environment-mac.yml
conda activate ai_model_server
python migrate.py
python server.py
