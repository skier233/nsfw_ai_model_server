# Define the URL for the latest Miniconda installer
$minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

# Define the path where the installer will be downloaded
$installerPath = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"

# Download the Miniconda installer
Write-Host "Downloading Miniconda installer..."
Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath

# Define the installation path
$installPath = "$env:LOCALAPPDATA\Miniconda3"

# Install Miniconda silently
Write-Host "Installing Miniconda..."
Start-Process -Wait -FilePath $installerPath -ArgumentList "/InstallationType=JustMe", "/AddToPath=0", "/RegisterPython=0", "/S", "/D=$installPath"

# Remove the installer after installation
Remove-Item $installerPath

# Initialize conda for PowerShell
Write-Host "Initializing conda for PowerShell..."
& "$installPath\Scripts\conda.exe" init powershell

# Update the current session with the new profile script
Write-Host "Updating profile script for the current session..."
$condaProfileScript = "$installPath\shell\condabin\conda-hook.ps1"
$profilePath = "$PROFILE"

if (!(Test-Path -Path $profilePath)) {
    # Create the profile file if it doesn't exist
    New-Item -ItemType File -Path $profilePath -Force
}

# Add the conda initialization script to the PowerShell profile
Add-Content -Path $profilePath -Value "if (Test-Path $condaProfileScript) { . $condaProfileScript }"

# Reload the profile in the current session to activate conda
Write-Host "Reloading the profile..."
. $PROFILE

# Verify the installation
Write-Host "Verifying conda installation..."
conda --version

Write-Host "Miniconda installation and configuration completed successfully."
