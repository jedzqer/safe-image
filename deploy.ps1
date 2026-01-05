param(
  [switch]$SkipModel,
  [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
  param(
    [string]$FilePath,
    [string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: $FilePath $Arguments"
  }
}

try {
  $Root = $PSScriptRoot
  if (-not $Root) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
  }
  if (-not $Root) {
    $Root = (Get-Location).Path
  }
  $Root = (Resolve-Path $Root).Path

  $CondaDir = Join-Path $Root ".conda"
  $CondaExe = Join-Path $CondaDir "Scripts\conda.exe"
  $EnvDir = Join-Path $Root ".conda_env"
  $Installer = Join-Path $Root "Miniconda3-latest-Windows-x86_64.exe"
  $ModelDir = Join-Path $Root "NSFWdetector"
  $Requirements = Join-Path $Root "requirements.txt"

  if (-not (Test-Path $CondaExe)) {
    $Url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    Write-Host "Downloading Miniconda..."
    Invoke-WebRequest -Uri $Url -OutFile $Installer

    Write-Host "Installing Miniconda..."
    $Args = "/S /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /D=$CondaDir"
    $proc = Start-Process -FilePath $Installer -ArgumentList $Args -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
      throw "Miniconda installer failed with exit code $($proc.ExitCode)."
    }
  }

  if (-not (Test-Path $CondaExe)) {
    throw "conda.exe not found at $CondaExe"
  }

  $EnvPython = Join-Path $EnvDir "python.exe"
  if (-not (Test-Path $EnvPython)) {
    Write-Host "Creating conda environment..."
    Invoke-Checked $CondaExe @("create", "-y", "-p", $EnvDir, "python=3.10.8")
  }

  if (-not (Test-Path $Requirements)) {
    throw "requirements.txt not found: $Requirements"
  }

  Write-Host "Installing Python dependencies..."
  Invoke-Checked $CondaExe @("run", "-p", $EnvDir, "python", "-m", "pip", "install", "--upgrade", "pip")
  Invoke-Checked $CondaExe @("run", "-p", $EnvDir, "python", "-m", "pip", "install", "-r", $Requirements)

  if (-not $SkipModel) {
    if (-not (Test-Path $ModelDir)) {
      Write-Host "Downloading NSFW model..."
      Invoke-Checked $CondaExe @("run", "-p", $EnvDir, "python", "-m", "pip", "install", "huggingface_hub")
      $Cmd = "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Falconsai/nsfw_image_detection', local_dir='NSFWdetector', local_dir_use_symlinks=False); print('Model download complete.')"
      Invoke-Checked $CondaExe @("run", "-p", $EnvDir, "python", "-c", $Cmd)
    } else {
      Write-Host "Model directory exists, skipping download."
    }
  } else {
    Write-Host "SkipModel enabled, skipping model download."
  }

  $Dirs = @("images", "recycle_bin", "SafeNet", "no_detection", "input")
  foreach ($d in $Dirs) {
    $Path = Join-Path $Root $d
    if (-not (Test-Path $Path)) {
      New-Item -ItemType Directory -Path $Path | Out-Null
    }
  }

  if (-not $SkipIndex) {
    $IndexScript = Join-Path $Root "index_img.py"
    if (-not (Test-Path $IndexScript)) {
      throw "index_img.py not found: $IndexScript"
    }
    Write-Host "Running index_img.py..."
    Invoke-Checked $CondaExe @("run", "-p", $EnvDir, "python", $IndexScript)
  } else {
    Write-Host "SkipIndex enabled, skipping index_img.py."
  }

  $AppScript = Join-Path $Root "app.py"
  if (-not (Test-Path $AppScript)) {
    throw "app.py not found: $AppScript"
  }
  Write-Host "Starting app.py..."
  Invoke-Checked $CondaExe @("run", "-p", $EnvDir, "python", $AppScript)
}
catch {
  Write-Error $_
  Read-Host "Press Enter to exit"
  exit 1
}
