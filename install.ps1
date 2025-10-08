conda deactivate
conda remove -n ai_model_server --all
conda env create -f environment-windows.yml
conda activate ai_model_server
python scripts/install_ffmpeg_cuda.py
python migrate.py
python server.py