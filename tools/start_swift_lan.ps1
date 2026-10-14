param(
    [string]$ConfigName = 'strata-swift-huihui-iq3_xxs.json',
    [string]$ExpectedGpu = '',
    [string]$ExpectedEngineVersion = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$envPath = Join-Path $root '.env'
$configPath = Join-Path $root $ConfigName
$python = Join-Path $root '.venv\Scripts\python.exe'
$server = Join-Path $root 'serve\server.py'

if (-not (Test-Path -LiteralPath $envPath)) { throw "Missing $envPath" }
if (-not (Test-Path -LiteralPath $configPath)) { throw "Missing $configPath" }
if (-not (Test-Path -LiteralPath $python)) { throw "Missing $python" }

$settings = @{}
foreach ($line in [IO.File]::ReadAllLines($envPath)) {
    if ($line -match '^([A-Z][A-Z0-9_]*)=(.*)$') {
        $settings[$Matches[1]] = $Matches[2]
    }
}
$bindAddress = $settings['STRATA_LAN_HOST']
$apiKey = $settings['STRATA_API_KEY']
if (-not $bindAddress -or -not $apiKey -or $apiKey.Length -lt 32) {
    throw 'STRATA_LAN_HOST and a long STRATA_API_KEY are required in .env'
}
$parsed = [Net.IPAddress]::None
if (-not [Net.IPAddress]::TryParse($bindAddress, [ref]$parsed) -or
    $parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
    $bindAddress -eq '127.0.0.1') {
    throw 'STRATA_LAN_HOST must be 0.0.0.0 or a LAN IPv4 address on this PC'
}
if ($bindAddress -ne '0.0.0.0' -and
    -not (Get-NetIPAddress -AddressFamily IPv4 -IPAddress $bindAddress -ErrorAction SilentlyContinue)) {
    throw "STRATA_LAN_HOST $bindAddress is not assigned to this PC"
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.host -ne '127.0.0.1' -or -not $config.port) {
    throw 'The base model config must retain a loopback host and port'
}

if ($config.backend -ne 'hip' -or -not (Test-Path -LiteralPath $config.exe -PathType Leaf)) {
    throw 'The HIP engine named by the model config is missing'
}
foreach ($libDir in $config.lib_dirs) {
    if (-not (Test-Path -LiteralPath $libDir -PathType Container)) {
        throw "Engine library directory is missing: $libDir"
    }
}
if ($config.vision -and -not (Test-Path -LiteralPath $config.vision.exe -PathType Leaf)) {
    throw 'The vision encoder named by the model config is missing'
}
if ($ExpectedEngineVersion) {
    $buildPath = Join-Path (Split-Path -Parent $config.exe) 'BUILD.json'
    if (-not (Test-Path -LiteralPath $buildPath -PathType Leaf) -or
        (Get-Content -LiteralPath $buildPath -Raw | ConvertFrom-Json).version -ne $ExpectedEngineVersion) {
        throw "Expected Strata engine $ExpectedEngineVersion is not installed at $($config.exe)"
    }
}
if ($ExpectedGpu) {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -eq $ExpectedGpu } | Select-Object -First 1
    if (-not $gpu) { throw "Expected GPU $ExpectedGpu is not present" }
}
if ($CheckOnly) {
    Write-Host "LAN launcher ready: $($config.model_name), engine $ExpectedEngineVersion, GPU $ExpectedGpu, port $($config.port)"
    exit 0
}

$env:STRATA_API_KEY = $apiKey
try {
    Set-Location -LiteralPath $root
    & $python -u $server --engine strata --config $configPath --host $bindAddress --port $config.port
    exit $LASTEXITCODE
} finally {
    Remove-Item Env:STRATA_API_KEY -ErrorAction SilentlyContinue
}
