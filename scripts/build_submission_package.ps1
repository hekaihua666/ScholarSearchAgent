param(
    [string]$OutputName = "ScholarSearchAgent_Onsite_20260830.zip"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dist = Join-Path $Root "dist"
$Stage = Join-Path $Dist "研问智搜_ScholarSearchAgent"
$Zip = Join-Path $Dist $OutputName

New-Item -ItemType Directory -Force $Dist | Out-Null
if (Test-Path $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Force $Stage | Out-Null

# Package exactly the version-controlled deliverables plus safe untracked files.
$files = & git -C $Root -c core.quotePath=false ls-files --cached --others --exclude-standard
foreach ($relative in $files) {
    if (-not $relative -or $relative -like "dist/*") { continue }
    if ($relative -eq ".env" -or $relative -like "*.key") {
        throw "Secret file refused: $relative"
    }
    $source = Join-Path $Root $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $target = Join-Path $Stage $relative
    New-Item -ItemType Directory -Force (Split-Path $target -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
}

# Fail closed if a real value was accidentally placed in the environment template.
$envTemplate = Join-Path $Stage ".env.example"
if (Test-Path $envTemplate) {
    $unsafe = Get-Content $envTemplate | Where-Object {
        $_ -match '^[A-Z0-9_]+\s*=\s*[^#\s]+' -and
        $_ -notmatch 'your_.*_here' -and $_ -notmatch '^#'
    }
    if ($unsafe) { throw "Non-placeholder value found in .env.example" }
}

if (Test-Path $Zip) { Remove-Item -LiteralPath $Zip -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -CompressionLevel Optimal

$hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = [ordered]@{
    package = $Zip
    sha256 = $hash
    created_at = (Get-Date).ToString("s")
    file_count = @($files).Count
    secrets_included = $false
    frozen_demo = "demo/frozen_response.json"
}
$manifest | ConvertTo-Json | Set-Content (Join-Path $Dist "package_manifest.json") -Encoding UTF8
$manifest | ConvertTo-Json
