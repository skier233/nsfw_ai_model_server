$ErrorActionPreference = "Stop"

if (-not $PSScriptRoot) {
    $PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Function to get the latest release version from GitHub
function Get-LatestReleaseVersion {
    $apiUrl = "https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
    $latestRelease = Invoke-RestMethod -Uri $apiUrl
    return $latestRelease.tag_name
}

# Function to read the local version from config/version.yaml
function Get-LocalVersion {
    $configPath = Join-Path $RepoRoot "config/version.yaml"
    if (-not (Test-Path $configPath)) {
        return $null
    }
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match 'VERSION:\s*(\S+)') {
        return $matches[1]
    }
    return $null
}

Push-Location $RepoRoot
try {
    $latestVersion = Get-LatestReleaseVersion
    $localVersion = Get-LocalVersion
    if ($latestVersion -eq $localVersion -and $localVersion) {
        Write-Host "You are up to date. Current version: $localVersion"
        return
    }

    $download = "https://github.com/skier233/nsfw_ai_model_server/releases/latest/download/nsfw_ai_model_server.zip"

    $zip = "nsfw_ai_model_server.zip"
    $tempDir = "temp_nsfw_ai_model_server"

    Write-Host "Downloading latest release"
    Invoke-WebRequest $download -OutFile $zip

    Write-Host "Extracting release files to temporary directory"
    Expand-Archive $zip -DestinationPath $tempDir -Force

    # Delete /config/config.yaml if it exists in the temporary directory
    $configFilePath = Join-Path $tempDir "config/config.yaml"
    if (Test-Path $configFilePath) {
        Write-Host "Deleting config.yaml from temporary directory to preserve existing configuration"
        Remove-Item $configFilePath -Force
    }

    $activeFilePath = Join-Path $tempDir "config/active_ai.yaml"
    if (Test-Path $activeFilePath) {
        Write-Host "Deleting active_ai.yaml from temporary directory to preserve existing configuration"
        Remove-Item $activeFilePath -Force
    }

    Write-Host "Copying files to the repository root"
    $tempDirFull = (Resolve-Path $tempDir).Path
    Get-ChildItem $tempDirFull -Recurse | ForEach-Object {
        if ($_.PSIsContainer) {
            return
        }

        $relativePath = $_.FullName.Substring($tempDirFull.Length).TrimStart([char[]]"\\/")
        $targetPath = Join-Path $RepoRoot $relativePath

        $targetDir = Split-Path $targetPath -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }

        Copy-Item $_.FullName -Destination $targetPath -Force
    }

    Write-Host "Cleaning up"
    Remove-Item $zip -Force
    Remove-Item $tempDir -Recurse -Force

    Write-Host "Update complete"

    conda activate ai_model_server
    pip uninstall ai-processing -y

    $environmentCandidates = @(
        (Join-Path $RepoRoot "environment-windows.yml"),
        (Join-Path $PSScriptRoot "environment-windows.yml")
    )
    $environmentFile = $environmentCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $environmentFile) {
        throw "Environment definition file 'environment-windows.yml' not found in repo root or install directory."
    }

    conda env update --file $environmentFile
}
finally {
    Pop-Location
}