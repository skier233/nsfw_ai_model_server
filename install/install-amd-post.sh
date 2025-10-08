#!/bin/bash
echo "Running AMD-specific Post-Install Steps"
conda activate ai_model_server
# Follow https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm ${location}/torch/lib/libhsa-runtime64.so*
conda install -c conda-forge gcc=12.1.0