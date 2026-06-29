param(
    [string]$InstallRoot = "C:\ai",
    [string]$EnvName = "diffusion-pipe",
    [switch]$SkipTorch,
    [switch]$SkipSubmodules
)

$ErrorActionPreference = "Stop"

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Warn($Message) {
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Require-Command($Command, $InstallHint) {
    $found = Get-Command $Command -ErrorAction SilentlyContinue
    if (-not $found) {
        throw "$Command was not found. $InstallHint"
    }
}

function Run($File, [string[]]$Args) {
    Write-Host ("$File " + ($Args -join " ")) -ForegroundColor DarkGray
    & $File @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $File $($Args -join ' ')"
    }
}

function WslPath($WindowsPath) {
    $result = & wsl.exe wslpath -a "$WindowsPath"
    if ($LASTEXITCODE -ne 0) {
        throw "wslpath failed for $WindowsPath"
    }
    return ($result | Select-Object -First 1).Trim()
}

function WslRun($Command) {
    Run "wsl.exe" @("-e", "bash", "-lc", $Command)
}

Step "Checking Windows prerequisites"
Require-Command "wsl.exe" "Install WSL2 first: wsl --install"
try {
    Run "wsl.exe" @("-e", "bash", "-lc", "echo WSL OK")
} catch {
    throw "WSL is installed, but no working Linux distro answered. Open PowerShell as admin and run: wsl --install"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoWsl = WslPath $RepoRoot
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$MambaRootWin = Join-Path $InstallRoot "micromamba"
$MambaRootWsl = WslPath $MambaRootWin
$Mamba = "$MambaRootWsl/bin/micromamba"
$EnvPrefix = "$MambaRootWsl/envs/$EnvName"

Write-Host "diffusion-pipe easy install"
Write-Host "Repo:        $RepoRoot"
Write-Host "InstallRoot: $InstallRoot"
Write-Host "Env:         $EnvPrefix"

if (-not (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
    Write-Host "  NVIDIA driver not found." -ForegroundColor Red
    Write-Host "  Install the latest driver before training:" -ForegroundColor Red
    Write-Host "  https://www.nvidia.com/drivers" -ForegroundColor Yellow
    Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
    Write-Host ""
    throw "NVIDIA driver required. Please install it and re-run."
} else {
    $gpuInfo = & nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    Write-Host $gpuInfo

    # CUDA 13.0 (PyTorch nightly cu130) requires driver >= 570.00
    $minDriver = 570
    $driverLine = ($gpuInfo -split "`n")[0]
    if ($driverLine -match ",\s*([\d]+)\.([\d]+),") {
        $driverMajor = [int]$Matches[1]
        if ($driverMajor -lt $minDriver) {
            Write-Host ""
            Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
            Write-Host "  Driver version $driverMajor.$($Matches[2]) is too old." -ForegroundColor Red
            Write-Host "  PyTorch cu130 requires driver 570.00 or newer." -ForegroundColor Red
            Write-Host "  Please update your NVIDIA driver and re-run:" -ForegroundColor Red
            Write-Host "  https://www.nvidia.com/drivers" -ForegroundColor Yellow
            Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
            Write-Host ""
            throw "Driver too old (found $driverMajor, need >= $minDriver). Please update and re-run."
        } else {
            Write-Host "Driver $driverMajor.$($Matches[2]) OK (>= $minDriver required for cu130)." -ForegroundColor Green
        }
    } else {
        Warn "Could not parse driver version. Continuing — if CUDA fails, update your driver to 570+ from https://www.nvidia.com/drivers"
    }
}

Step "Preparing repository"
if (-not $SkipSubmodules) {
    if (Test-Path (Join-Path $RepoRoot ".git")) {
        Require-Command "git.exe" "Install Git for Windows first, or rerun with -SkipSubmodules."
        Run "git.exe" @("-C", $RepoRoot, "submodule", "update", "--init", "--recursive")
    } else {
        Warn "This folder is not a git checkout; skipping submodule update."
    }
}

Step "Installing micromamba in WSL path"
New-Item -ItemType Directory -Force -Path $MambaRootWin | Out-Null
WslRun "mkdir -p '$MambaRootWsl/bin' '$MambaRootWsl/root' '$MambaRootWsl/envs'"
WslRun "if [ ! -x '$Mamba' ]; then curl -L 'https://micro.mamba.pm/api/micromamba/linux-64/latest' | tar -xj -C '$MambaRootWsl' --strip-components=1 bin/micromamba; fi"
WslRun "'$Mamba' --version"

Step "Creating Python environment"
WslRun "export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'; if [ ! -x '$EnvPrefix/bin/python' ]; then '$Mamba' create -y -p '$EnvPrefix' -c conda-forge python=3.12 pip; else echo 'Environment already exists: $EnvPrefix'; fi"

Step "Writing local run_wsl environment"
$localEnv = @"
export DIFFUSION_PIPE_MAMBA_ROOT='$MambaRootWsl'
export DIFFUSION_PIPE_ENV_PREFIX='$EnvPrefix'
"@
Set-Content -LiteralPath (Join-Path $RepoRoot ".diffusion_pipe_env") -Value $localEnv -Encoding ASCII

Step "Upgrading pip tooling"
WslRun "export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'; '$Mamba' run -p '$EnvPrefix' python -m pip install --upgrade pip wheel setuptools packaging ninja"

if (-not $SkipTorch) {
    Step "Installing PyTorch CUDA packages"
    $torchCmd = @"
set -e
export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'
'$Mamba' run -p '$EnvPrefix' python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
'$Mamba' run -p '$EnvPrefix' python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
"@
    WslRun $torchCmd
}

Step "Installing diffusion-pipe requirements"
WslRun "export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'; cd '$RepoWsl'; '$Mamba' run -p '$EnvPrefix' python -m pip install -r requirements.txt"

Step "Installing local UI dependencies"
WslRun "export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'; '$Mamba' run -p '$EnvPrefix' python -m pip install gradio"

Step "Installing CUDA nvcc package if needed"
WslRun "export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'; '$Mamba' run -p '$EnvPrefix' python -m pip install nvidia-cuda-nvcc"

Step "Verifying DeepSpeed, Torch, TensorBoard, and Gradio"
$verify = @"
set -e
export MAMBA_ROOT_PREFIX='$MambaRootWsl/root'
cd '$RepoWsl'
'$Mamba' run -p '$EnvPrefix' python - <<'PY'
import importlib
for name in ['torch', 'deepspeed', 'torchaudio', 'torchvision', 'tensorboard', 'gradio']:
    mod = importlib.import_module(name)
    print(name, getattr(mod, '__version__', 'ok'))
import torch
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
"@
WslRun $verify

Step "Writing quick start note"
$note = @"
diffusion-pipe easy install complete.

Run these from Windows:
  dp_ui.bat              Web UI for config/run/TensorBoard
  dp_wizard.bat          Multi-model config wizard
  dp_run.bat             Run existing generated configs
  dp_tensorboard.bat     TensorBoard launcher

If WSL paths look odd, Windows paths like "C:\AI\datasets\name" and "M:\Models\..." are accepted by the helper scripts.
"@
$notePath = Join-Path $RepoRoot "INSTALL_DONE.txt"
Set-Content -LiteralPath $notePath -Value $note -Encoding ASCII
Write-Host $note

Write-Host ""
Write-Host "Done. Start the UI with: $RepoRoot\dp_ui.bat" -ForegroundColor Green
