#!/bin/bash

main() {
    local previous_opts
    previous_opts=$(set +o)
    set -euo pipefail

    if ! command -v unzip &> /dev/null; then
        echo "unzip could not be found, run sudo apt-get install unzip to install it."
        eval "$previous_opts"
        return 1
    fi

    local script_dir repo_root
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    repo_root="$(cd -- "$script_dir/.." && pwd -P)"

    pushd "$repo_root" > /dev/null
    trap 'popd > /dev/null' RETURN

    get_latest_release_version() {
        local apiUrl="https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
        curl -s "$apiUrl" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'
    }

    get_local_version() {
        local configPath="$repo_root/config/version.yaml"
        if [[ -f "$configPath" ]]; then
            grep 'VERSION:' "$configPath" | awk '{print $2}'
        else
            echo "null"
        fi
    }

    local latestVersion localVersion
    latestVersion=$(get_latest_release_version)
    localVersion=$(get_local_version)

    if [[ "$latestVersion" == "$localVersion" ]]; then
        echo "You are up to date. Current version: $localVersion"
        eval "$previous_opts"
        return 0
    fi

    echo "Updating to latest version: $latestVersion from $localVersion"

    local download="https://github.com/skier233/nsfw_ai_model_server/releases/latest/download/nsfw_ai_model_server.zip"
    local zip="nsfw_ai_model_server.zip"
    local tempDir="temp_nsfw_ai_model_server"

    echo "Downloading latest release"
    wget -O "$zip" "$download"

    echo "Extracting release files to temporary directory"
    mkdir -p "$tempDir"
    unzip -o "$zip" -d "$tempDir"

    local configFilePath="$tempDir/config/config.yaml"
    if [[ -f "$configFilePath" ]]; then
        echo "Deleting config.yaml from temporary directory to preserve existing configuration"
        rm -f "$configFilePath"
    fi

    local activeFilePath="$tempDir/config/active_ai.yaml"
    if [[ -f "$activeFilePath" ]]; then
        echo "Deleting active_ai.yaml from temporary directory to preserve existing configuration"
        rm -f "$activeFilePath"
    fi

    echo "Copying files to the repository root"
    rsync -av --exclude='update.sh' "$tempDir"/* ./

    echo "Cleaning up"
    rm -v -f "$zip"
    rm -v -rf "$tempDir"

    echo "Update complete"

    conda activate ai_model_server
    pip uninstall ai-processing -y || true

    local target_profile="${1:-}"
    local env_file
    case "$target_profile" in
        --amd)
            env_file="$script_dir/environment-linux-amd.yml"
            conda env update --file "$env_file"
            source "$script_dir/install-amd-post.sh"
            ;;
        --intel)
            env_file="$script_dir/environment-linux-intel.yml"
            conda env update --file "$env_file"
            ;;
        --mac)
            env_file="$script_dir/environment-mac.yml"
            conda env update --file "$env_file"
            ;;
        "")
            env_file="$script_dir/environment-linux.yml"
            conda env update --file "$env_file"
            ;;
        *)
            echo "Unknown flag '$target_profile'. Supported: --amd, --intel, --mac"
            eval "$previous_opts"
            return 1
            ;;
    esac

    eval "$previous_opts"
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
    exit $?
else
    main "$@"
    return $?
fi