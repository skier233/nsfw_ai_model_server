Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
	Write-Error 'Conda is not installed or not initialized for PowerShell. Please run .\install-conda.ps1 first.'
}

Push-Location $PSScriptRoot

try {
	try { conda deactivate | Out-Null } catch { }

	$envName = 'ai_model_server'
	$envList = conda env list | Out-String
	if ($envList -match "^$envName\s") {
		Write-Output "Removing existing conda environment '$envName'..."
		conda remove -n $envName --all -y | Out-Null
	}

	Write-Output 'Creating conda environment (environment-windows.yml)...'
	conda env create -f (Join-Path $PSScriptRoot 'environment-windows.yml')

	Write-Output "Activating environment '$envName'..."
	conda activate $envName

	Write-Output 'Installing FFmpeg with CUDA/NVDEC support...'
	python (Join-Path $PSScriptRoot 'scripts/install_ffmpeg_cuda.py')

	Write-Output 'Running migration'
	python (Join-Path $PSScriptRoot 'migrate.py')

	Write-Output 'Starting server...'
	python (Join-Path $PSScriptRoot 'server.py')
}
finally {
	Pop-Location
}