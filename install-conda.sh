#!/bin/bash

# Define the URL for the latest Miniconda installer
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Define the path where the installer will be downloaded
INSTALLER_PATH="$HOME/Miniconda3-latest-Linux-x86_64.sh"

# Download the Miniconda installer
echo "Downloading Miniconda installer..."
curl -o $INSTALLER_PATH $MINICONDA_URL

# Define the installation path
INSTALL_PATH="$HOME/miniconda3"

# Install Miniconda silently
echo "Installing Miniconda..."
bash $INSTALLER_PATH -b -p $INSTALL_PATH

# Remove the installer after installation
rm $INSTALLER_PATH

# Initialize conda for bash
echo "Initializing conda for bash..."
$INSTALL_PATH/bin/conda init bash

# Reload the profile in the current session to activate conda
echo "Reloading the profile..."
source "$HOME/.bashrc"

# Verify the installation
echo "Verifying conda installation..."
conda --version

echo "Miniconda installation and configuration completed successfully."
