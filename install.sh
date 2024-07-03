#!/bin/bash

# Activate conda environment and start server
conda env create -f environment-linux.yml
conda activate ai_model_server
python server.py