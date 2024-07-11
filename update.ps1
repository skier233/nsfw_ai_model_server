# Ensure the PSYaml module is installed
if (-not (Get-Module -ListAvailable -Name PSYaml)) {
    Install-Module -Name PSYaml -Force -Scope CurrentUser
}

Import-Module PSYaml

# Function to get the latest release version from GitHub
function Get-LatestReleaseVersion {
    $apiUrl = "https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
    $latestRelease = Invoke-RestMethod -Uri $apiUrl
    return $latestRelease.tag_name
}

# Function to read the local version from config.yaml
function Get-LocalVersion {
    $configPath = "./config/config.yaml"
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match 'VERSION:\s*(\S+)') {
        return $matches[1]
    }
    return $null
}

function Load-YamlFile {
    param (
        [string]$filePath
    )
    return ConvertFrom-Yaml (Get-Content $filePath -Raw)
}

function Save-YamlFile {
    param (
        [hashtable]$data,
        [string]$filePath
    )
    $yamlContent = ConvertTo-Yaml $data
    Set-Content -Path $filePath -Value $yamlContent
}

function Merge-Yaml {
    param (
        [hashtable]$userConfig,
        [hashtable]$newConfig
    )
    foreach ($key in $newConfig.Keys) {
        if ($userConfig.ContainsKey($key)) {
            if ($userConfig[$key] -is [hashtable] -and $newConfig[$key] -is [hashtable]) {
                $userConfig[$key] = Merge-Yaml -userConfig $userConfig[$key] -newConfig $newConfig[$key]
            } elseif ($userConfig[$key] -is [array] -and $newConfig[$key] -is [array]) {
                $userConfig[$key] += $newConfig[$key] | Where-Object { $_ -notin $userConfig[$key] }
            } else {
                if ($userConfig[$key] -notmatch '# preserve') {
                    $userConfig[$key] = $newConfig[$key]
                }
            }
        } else {
            $userConfig[$key] = $newConfig[$key]
        }
    }
    return $userConfig
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


$localConfigDir = "./config"
$newConfigDir = "$tempDir/config"
# Get all local yaml files
$localYamlFiles = Get-ChildItem -Path $localConfigDir -Filter *.yaml -Recurse

foreach ($localYamlFile in $localYamlFiles) {
    $relativePath = $localYamlFile.FullName.Substring((Get-Location).Path.Length + $localConfigDir.Length + 1)
    $newYamlFile = Join-Path $newConfigDir $relativePath

    if (Test-Path $newYamlFile) {
        Write-Host "Merging $localYamlFile with $newYamlFile"
        $userConfig = Load-YamlFile -filePath $localYamlFile.FullName
        $newConfig = Load-YamlFile -filePath $newYamlFile

        $mergedConfig = Merge-Yaml -userConfig $userConfig -newConfig $newConfig

        Save-YamlFile -data $mergedConfig -filePath $localYamlFile.FullName
    } else {
        Write-Host "New configuration file $newYamlFile does not exist. Skipping."
    }
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