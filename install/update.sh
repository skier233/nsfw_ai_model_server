#!/bin/bash

if ! command -v unzip &> /dev/null; then
    echo "unzip could not be found, run sudo apt-get install unzip to install it."
    exit 1
fi

# Function to get the latest release version from GitHub
get_latest_release_version() {
    apiUrl="https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
    latestRelease=$(curl -s $apiUrl | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
    echo $latestRelease
}

# Function to read the local version from config.yaml
get_local_version() {
    configPath="../config/version.yaml"
    if [[ -f "$configPath" ]]; then
        localVersion=$(grep 'VERSION:' $configPath | awk '{print $2}')
        echo $localVersion
    else
        echo "null"
    fi
}

latestVersion=$(get_latest_release_version)
localVersion=$(get_local_version)

if [[ "$latestVersion" == "$localVersion" ]]; then
    echo "You are up to date. Current version: $localVersion"
    exit 0
fi

echo "Updating to latest version: $latestVersion from $localVersion"

download="https://github.com/skier233/nsfw_ai_model_server/releases/latest/download/nsfw_ai_model_server.zip"
zip="nsfw_ai_model_server.zip"
tempDir="temp_nsfw_ai_model_server"

echo "Downloading latest release"
wget -O $zip $download

echo "Extracting release files to temporary directory"
mkdir -p $tempDir
unzip -o $zip -d $tempDir

# Delete /config/config.yaml if it exists in the temporary directory
configFilePath="$tempDir/config/config.yaml"
if [ -f "$configFilePath" ]; then
    echo "Deleting config.yaml from temporary directory to preserve existing configuration"
    rm -f "$configFilePath"
fi

activeFilePath="$tempDir/config/active_ai.yaml"
if [ -f "$activeFilePath" ]; then
    echo "Deleting active_ai.yaml from temporary directory to preserve existing configuration"
    rm -f "$activeFilePath"
fi

echo "Copying files to the current directory"
rsync -av --exclude='update.sh' $tempDir/* . || { echo "Failed to copy files."; exit 1; }
echo "Cleaning up"
rm -v -f $zip || { echo "Failed to remove the zip file."; exit 1; }
rm -v -rf $tempDir || { echo "Failed to remove the temporary directory."; exit 1; }

echo "Update complete"

# Activate conda environment and update
conda activate ai_model_server
pip uninstall ai-processing -y
if [ "$1" = '--amd' ]; then
    conda env update --file ./environment-linux-amd.yml
    source ./install-amd-post.sh
elif [ "$1" = '--intel' ]; then
    conda env update --file ./environment-linux-intel.yml
elif [ "$1" = '--mac' ]; then
    conda env update --file ./environment-mac.yml
else
    conda env update --file ./environment-linux.yml
fi