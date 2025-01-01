conda deactivate
conda remove -n ai_model_server --all
conda env create -f environment-windows.yml
conda activate ai_model_server
python migrate.py
python server.py