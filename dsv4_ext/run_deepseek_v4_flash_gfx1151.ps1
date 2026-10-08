param(
    [string]$ModelDir = 'C:\models\DeepSeek-V4-Flash-UD-IQ1_M\UD-IQ1_M',
    [int]$Port = 8095,
    [int]$Context = 512,
    [int]$HostThreads = 4,
    [int]$CpuThreads = 16,
    [ValidateSet('fixed', 'shape')]
    [string]$MatvecPolicy = 'shape',
    [string]$RocmVenv = 'C:\Dev\Strata-native-hip\.rocm-win',
    [string]$PythonExe = 'C:\Users\HarutoWatanabe\.unsloth\studio\unsloth_studio\Scripts\python.exe',
    [string]$BuildDir = (Join-Path $PSScriptRoot 'build-shape32-v256-gfx1151'),
    [string]$WorkDir = 'C:\models\DeepSeek-V4-Flash-UD-IQ1_M\runs\serve-gfx1151'
)
$ErrorActionPreference = 'Stop'
$firstShard = Join-Path $ModelDir 'DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf'
$engine = Join-Path $BuildDir 'dsv4_run.exe'
$python = $PythonExe
$rocmSdk = Join-Path $RocmVenv 'Scripts\rocm-sdk.exe'
$server = Join-Path $PSScriptRoot 'tools\serve_dsv4.py'
foreach ($required in @($firstShard, $engine, $python, $rocmSdk, $server)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}
$rocmOutput = & $rocmSdk path --root
$rocmExit = $LASTEXITCODE
$rocmRoot = ($rocmOutput | Select-Object -First 1).Trim()
if ($rocmExit -ne 0 -or -not $rocmRoot) { throw 'rocm-sdk did not return the ROCm root' }
$libDir = Join-Path $rocmRoot 'bin'
$arguments = @(
    $server, '--hip', '--model', $firstShard, '--exe', $engine, '--lib-dir', $libDir,
    '--host', '127.0.0.1', '--port', "$Port", '--ctx', "$Context", '--threads', "$HostThreads", '--workdir', $WorkDir,
    '--expert-cache', '0', '--expert-vram-reserve-mib', '4096', '--expert-ram', 'profile', '--ram-cache-mib', '0',
    "--engine-arg=--cpu-experts 6", "--engine-arg=--cpu-threads $CpuThreads",
    '--engine-arg=--cpu-moe-kernel native', '--engine-arg=--fused-attention-proj',
    '--engine-env', "OMP_NUM_THREADS=$HostThreads", '--engine-env', 'OMP_WAIT_POLICY=PASSIVE', '--engine-env', 'KMP_BLOCKTIME=0',
    '--engine-env', "DSV4_MATVEC_POLICY=$MatvecPolicy", '--engine-env', 'DSV4_CPU_IQ1M_KERNEL=auto'
)
Write-Host "Starting DeepSeek V4 Flash on http://127.0.0.1:$Port/ using $engine"
Write-Host 'The server remains in this terminal; press Ctrl+C to stop it.'
& $python @arguments
exit $LASTEXITCODE
