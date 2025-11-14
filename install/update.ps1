# Function to get the latest release version from GitHub
function Get-LatestReleaseVersion {
    $apiUrl = "https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
    $latestRelease = Invoke-RestMethod -Uri $apiUrl
    return $latestRelease.tag_name
}

# Function to read the local version from config.yaml
function Get-LocalVersion {
    $configPath = "../config/version.yaml"
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match 'VERSION:\s*(\S+)') {
        return $matches[1]
    }
    return $null
}

$latestVersion = Get-LatestReleaseVersion
$localVersion = Get-LocalVersion
if ($latestVersion -eq $localVersion) {
    Write-Host "You are up to date. Current version: $localVersion"
    exit
}

$download = "https://github.com/skier233/nsfw_ai_model_server/releases/latest/download/nsfw_ai_model_server.zip"

$zip = "nsfw_ai_model_server.zip"
$tempDir = "temp_nsfw_ai_model_server"

Write-Host "Downloading latest release"
Invoke-WebRequest $download -OutFile $zip

Write-Host "Extracting release files to temporary directory"
Expand-Archive $zip -DestinationPath $tempDir -Force

# Delete /config/config.yaml if it exists in the temporary directory
$configFilePath = Join-Path $tempDir "config\config.yaml"
if (Test-Path $configFilePath) {
    Write-Host "Deleting config.yaml from temporary directory to preserve existing configuration"
    Remove-Item $configFilePath -Force
}

$activeFilePath = Join-Path $tempDir "config\active_ai.yaml"
if (Test-Path $activeFilePath) {
    Write-Host "Deleting active_ai.yaml from temporary directory to preserve existing configuration"
    Remove-Item $activeFilePath -Force
}

Write-Host "Copying files to the current directory"
Get-ChildItem $tempDir -Recurse | ForEach-Object {
    # Calculate the relative path from the temp directory
    $relativePath = $_.FullName.Substring((Get-Location).Path.Length + $tempDir.Length + 2)
    $targetPath = Join-Path (Get-Location) $relativePath

    # Ensure the target directory exists
    $targetDir = Split-Path $targetPath -Parent
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    # Copy files, skipping directories since their contents are being copied individually
    if (-not $_.PSIsContainer) {
        Copy-Item $_.FullName -Destination $targetPath -Force
    }
}

Write-Host "Cleaning up"
Remove-Item $zip -Force
Remove-Item $tempDir -Recurse -Force

Write-Host "Update complete"

conda activate ai_model_server
pip uninstall ai-processing -y
conda env update --file .\environment-windows.yml