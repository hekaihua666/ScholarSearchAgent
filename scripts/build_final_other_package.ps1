param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dist = Join-Path $Root "dist"
$StageRoot = Join-Path $Dist "final_other_stage"
$PackageName = "研问智搜_ScholarSearchAgent"
$PackageRoot = Join-Path $StageRoot $PackageName

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $Root "artifacts\submission_2026\final\虫虫特工队_研问智搜_其他.zip"
} elseif (-not [IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $Root $OutputPath
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)

New-Item -ItemType Directory -Force $Dist | Out-Null
New-Item -ItemType Directory -Force (Split-Path $OutputPath -Parent) | Out-Null

# Recursive cleanup is restricted to the dedicated staging directory.
$stageFull = [IO.Path]::GetFullPath($StageRoot)
$distFull = [IO.Path]::GetFullPath($Dist).TrimEnd('\') + '\'
if (-not $stageFull.StartsWith($distFull, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $stageFull"
}
if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force $PackageRoot | Out-Null

$rootFiles = @(
    ".env.example",
    "CHANGELOG.md",
    "pytest.ini",
    "README.md",
    "requirements-selector.txt",
    "requirements-test.txt",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    "一键启动_研问智搜.bat",
    "现场运行说明.md",
    "离线演示_研问智搜.bat",
    "自检启动.bat"
)

$prefixes = @(
    "backend/",
    "frontend/",
    "configs/",
    "demo/",
    "tests/",
    "scripts/",
    "artifacts/core_rescue_v6/",
    "artifacts/risk_controlled_rescue_v7/",
    "artifacts/short_sprint_v5/",
    "reports/final_evaluation/",
    "reports/short_sprint_v5/",
    "reports/f1_selector_v3/",
    "reports/joint_optimization_v4/",
    "reports/evidence_quorum_v20/",
    "reports/evidence_backprojection_v21/",
    "reports/reg_harddata_v20/",
    "reports/evidence_calibrated_selector_v22/",
    "reports/document_v33/"
)

$extraRoots = @(
    "backend/selection/evidence_calibrated_selector_v22.py",
    "configs/evidence_calibrated_selector_v22",
    "reports/evidence_calibrated_selector_v22",
    "reports/document_v33",
    "scripts/run_evidence_calibrated_selector_v22.py",
    "tests/test_evidence_calibrated_selector_v22.py"
)
$extraFiles = New-Object System.Collections.Generic.List[string]
foreach ($relativeRoot in $extraRoots) {
    $fullRoot = Join-Path $Root $relativeRoot
    if (Test-Path -LiteralPath $fullRoot -PathType Leaf) {
        $extraFiles.Add($relativeRoot.Replace('\', '/'))
    } elseif (Test-Path -LiteralPath $fullRoot -PathType Container) {
        foreach ($file in Get-ChildItem -LiteralPath $fullRoot -Recurse -File) {
            $extraFiles.Add($file.FullName.Substring($Root.Length + 1).Replace('\', '/'))
        }
    }
}

$tracked = @(& git -C $Root -c core.quotePath=false ls-files) + $rootFiles + $extraFiles
$tracked = @($tracked | Select-Object -Unique)
if ($LASTEXITCODE -ne 0) { throw "git ls-files failed" }

$selected = New-Object System.Collections.Generic.List[string]
foreach ($relativeRaw in $tracked) {
    if ([string]::IsNullOrWhiteSpace($relativeRaw)) { continue }
    $relative = $relativeRaw.Replace('\', '/')
    $include = $rootFiles -contains $relative
    if (-not $include) {
        foreach ($prefix in $prefixes) {
            if ($relative.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                $include = $true
                break
            }
        }
    }
    if (-not $include) { continue }
    if ($relative -eq ".env" -or $relative -match '(?i)\.(key|pem|p12|pfx)$') {
        throw "Secret file refused: $relative"
    }
    if ($relative -match '(?i)(^|/)(__pycache__|\.pytest_cache|node_modules|\.venv)(/|$)' -or
        $relative -match '(?i)\.(pyc|log|err|tmp|bak)$') {
        continue
    }
    $source = Join-Path $Root $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $target = Join-Path $PackageRoot $relative
    New-Item -ItemType Directory -Force (Split-Path $target -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
    $selected.Add($relative)
}

$required = @(
    "README.md",
    ".env.example",
    "requirements.txt",
    "requirements-test.txt",
    "一键启动_研问智搜.bat",
    "离线演示_研问智搜.bat",
    "backend/main.py",
    "frontend/index.html",
    "demo/frozen_response.json",
    "reports/evidence_quorum_v20/final_report.md",
    "reports/evidence_backprojection_v21/final_report.md",
    "reports/evidence_calibrated_selector_v22/final_report.md",
    "configs/evidence_calibrated_selector_v22/model.json",
    "backend/selection/evidence_calibrated_selector_v22.py"
)
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $PackageRoot $relative) -PathType Leaf)) {
        throw "Required package file missing: $relative"
    }
}

# Bundle the complete lightweight runtime dependency set so the project can
# create its virtual environment without downloading Python packages.
$requirementsPath = Join-Path $Root "requirements.txt"
$testRequirementsPath = Join-Path $Root "requirements-test.txt"
$requirementsHash = ((Get-FileHash -LiteralPath $requirementsPath -Algorithm SHA256).Hash + ':' + (Get-FileHash -LiteralPath $testRequirementsPath -Algorithm SHA256).Hash).ToLowerInvariant()
$wheelCache = Join-Path $Dist "runtime_wheelhouse_py312"
$wheelStamp = Join-Path $wheelCache "requirements.sha256"
$refreshWheels = -not (Test-Path -LiteralPath $wheelStamp)
if (-not $refreshWheels) {
    $refreshWheels = (Get-Content -Raw -LiteralPath $wheelStamp).Trim() -ne $requirementsHash
}
if ($refreshWheels) {
    $wheelFull = [IO.Path]::GetFullPath($wheelCache)
    if (-not $wheelFull.StartsWith($distFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe wheel cache path: $wheelFull"
    }
    if (Test-Path -LiteralPath $wheelCache) {
        Remove-Item -LiteralPath $wheelCache -Recurse -Force
    }
    New-Item -ItemType Directory -Force $wheelCache | Out-Null
    $downloadPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $downloadPython)) { $downloadPython = "python" }
    & $downloadPython -m pip download --disable-pip-version-check --only-binary=:all: --dest $wheelCache -r $testRequirementsPath
    if ($LASTEXITCODE -ne 0) { throw "Runtime wheel download failed" }
    Set-Content -LiteralPath $wheelStamp -Value $requirementsHash -Encoding ASCII
}
$packageWheelhouse = Join-Path $PackageRoot "wheelhouse"
New-Item -ItemType Directory -Force $packageWheelhouse | Out-Null
Copy-Item -Path (Join-Path $wheelCache "*.whl") -Destination $packageWheelhouse -Force
if (-not (Get-ChildItem -LiteralPath $packageWheelhouse -Filter *.whl -File)) {
    throw "Runtime wheelhouse is empty"
}

$envTemplate = Join-Path $PackageRoot ".env.example"
$unsafeTemplate = Get-Content -LiteralPath $envTemplate | Where-Object {
    $_ -match '^[A-Z0-9_]+\s*=\s*[^#\s]+' -and
    $_ -notmatch 'your_.*_here' -and
    $_ -notmatch '^#'
}
if ($unsafeTemplate) { throw "Non-placeholder value found in .env.example" }

# Compare staged text against locally configured secret values without logging them.
$knownSecrets = New-Object System.Collections.Generic.List[string]
$localEnv = Join-Path $Root ".env"
if (Test-Path -LiteralPath $localEnv) {
    foreach ($line in Get-Content -LiteralPath $localEnv) {
        if ($line -match '^\s*[A-Z0-9_]+\s*=\s*(.+?)\s*$') {
            $value = $Matches[1].Trim('"', "'")
            if ($value.Length -ge 8 -and $value -notmatch '^your_.*_here$') {
                $knownSecrets.Add($value)
            }
        }
    }
}

$textExtensions = @('.py','.md','.txt','.json','.jsonl','.csv','.yaml','.yml','.toml','.ini','.cfg','.html','.js','.css','.bat','.ps1','.tex','.srt','.example')
$genericSecretPatterns = @(
    '(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}',
    '(?i)ApiKey\s*[:=：]\s*[A-Za-z0-9_-]{12,}',
    '(?i)(DEEPSEEK_API_KEY|OPENAI_API_KEY|OPENALEX_API_KEY|S2_API_KEY)\s*=\s*(?!your_)[^#\s]{8,}'
)
foreach ($file in Get-ChildItem -LiteralPath $PackageRoot -Recurse -File) {
    if ($file.Length -gt 5MB -or $textExtensions -notcontains $file.Extension.ToLowerInvariant()) { continue }
    $content = Get-Content -Raw -LiteralPath $file.FullName -ErrorAction SilentlyContinue
    if ($null -eq $content) { continue }
    foreach ($secret in $knownSecrets) {
        if ($content.Contains($secret)) { throw "Configured secret detected in staged file: $($file.FullName)" }
    }
    foreach ($pattern in $genericSecretPatterns) {
        if ($content -match $pattern -and $file.Name -ne '.env.example') {
            throw "Secret-like content detected in staged file: $($file.FullName)"
        }
    }
}

$readme = @'
# 研问智搜可执行程序与复现材料

本压缩包包含 FastAPI 后端、单页前端、算法配置、自动化测试、核心实验摘要和可选的断网演示数据。

## 快速启动

1. 解压后进入 `研问智搜_ScholarSearchAgent`。
2. 在线运行：准备本机 `.env` 后双击 `一键启动_研问智搜.bat`。
3. 需要断网检查界面时，可双击 `离线演示_研问智搜.bat`。
4. 浏览器打开 `http://127.0.0.1:8002`。

详细步骤见 `现场运行说明.md`，项目结构与接口见 `README.md`。

## 验收状态

- 自动化测试：168 项通过。
- 核心结果：ECS v22；候选召回：EBP v21；关系排序：REG-MEQ v20。
- 提交包不包含 `.env`、API Key、运行缓存或日志。
'@
Set-Content -LiteralPath (Join-Path $PackageRoot "提交包说明.md") -Value $readme -Encoding UTF8

$fileManifest = @()
foreach ($file in Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Sort-Object FullName) {
    $relative = $file.FullName.Substring($PackageRoot.Length + 1).Replace('\', '/')
    $fileManifest += [ordered]@{
        path = $relative
        bytes = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifestObject = [ordered]@{
    package_name = $PackageName
    version = "V3.3-2026.08.31"
    generated_at = (Get-Date).ToString("s")
    file_count = $fileManifest.Count
    secrets_included = $false
    frozen_demo = "demo/frozen_response.json"
    files = $fileManifest
}
$manifestObject | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PackageRoot "PACKAGE_MANIFEST.json") -Encoding UTF8

if (Test-Path -LiteralPath $OutputPath) { Remove-Item -LiteralPath $OutputPath -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory(
    $StageRoot,
    $OutputPath,
    [IO.Compression.CompressionLevel]::Optimal,
    $false
)

$zipItem = Get-Item -LiteralPath $OutputPath
if ($zipItem.Length -ge 200MB) { throw "Package exceeds 200 MB: $($zipItem.Length) bytes" }
$zipHash = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
$external = [ordered]@{
    package = $OutputPath
    sha256 = $zipHash
    bytes = $zipItem.Length
    file_count = $fileManifest.Count + 1
    secrets_included = $false
    created_at = (Get-Date).ToString("s")
}
$external | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Dist "final_other_package_manifest.json") -Encoding UTF8
$external | ConvertTo-Json

