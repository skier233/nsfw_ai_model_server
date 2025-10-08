$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
conda activate ai_model_server
python server.py
Pop-Location