#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ENV_NAME="ai_model_server"

if ! command -v conda >/dev/null 2>&1; then
	echo "Conda is not installed. Please run ./install-conda-mac.sh first." >&2
	exit 1
fi

export CONDA_SUBDIR=osx-arm64

eval "$(conda shell.bash hook)"

conda deactivate >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
	echo "Removing existing conda environment '$ENV_NAME'..."
	conda remove -n "$ENV_NAME" --all -y
fi

echo "Creating conda environment from environment-mac.yml..."
conda env create -f "$PROJECT_ROOT/environment-mac.yml"

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Running database migrations..."
python "$PROJECT_ROOT/migrate.py"

echo "Starting server..."
python "$PROJECT_ROOT/server.py"
