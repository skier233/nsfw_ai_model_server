#!/bin/bash

# Activate conda environment and start server
conda deactivate
conda remove -n ai_model_server --all
conda env create -f environment-linux.yml
conda activate ai_model_server
python server.py