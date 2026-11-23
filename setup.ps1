# Prepares this project on Windows. Run it from the project folder, in PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# It installs what's missing:
#   - uv, which installs Python 3.12 (for your user, no administrator needed)
#   - the AWS CLI v2, with winget (Windows asks for administrator approval)
#   - .venv, with requirements.txt (including the YellowDog CLI)
#   - .env, copied from .env.example if you don't have one
#
# Safe to re-run: anything already installed is left alone.

$ErrorActionPreference = "Stop"
$UvVersion = "0.13.7"
$PythonVersion = "3.12"
Set-Location $PSScriptRoot

function Write-Step($Text) { Write-Host "`n=== $Text" -ForegroundColor Cyan }
function Test-Command($Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
function Assert-Success($What) { if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" } }
function Add-SessionPath($Folder) { if ((Test-Path $Folder) -and ($env:Path -notlike "*$Folder*")) { $env:Path = "$Folder;$env:Path" } }

# Where uv and the AWS CLI install to, so this session finds them straight after.
Add-SessionPath (Join-Path $HOME ".local\bin")
Add-SessionPath "C:\Program Files\Amazon\AWSCLIV2"

Write-Step "uv $UvVersion (installs Python)"
if (Test-Command uv) {
    Write-Host "already installed: $(uv --version)"
} else {
    $installer = Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1"
    & ([scriptblock]::Create($installer))
    Add-SessionPath (Join-Path $HOME ".local\bin")
}

Write-Step "Python $PythonVersion"
uv python install --no-bin $PythonVersion; Assert-Success "Installing Python"

Write-Step "AWS CLI v2"
if (Test-Command aws) {
    Write-Host "already installed: $(aws --version)"
} elseif (Test-Command winget) {
    winget install --exact --id Amazon.AWSCLI --accept-package-agreements --accept-source-agreements
    Assert-Success "Installing the AWS CLI"
    Add-SessionPath "C:\Program Files\Amazon\AWSCLIV2"
} else {
    $msi = Join-Path $env:TEMP "AWSCLIV2.msi"
    Invoke-WebRequest "https://awscli.amazonaws.com/AWSCLIV2.msi" -OutFile $msi
    Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /passive" -Wait
    Remove-Item $msi
    Add-SessionPath "C:\Program Files\Amazon\AWSCLIV2"
}

Write-Step "Project environment, .venv"
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "already exists: $(& .venv\Scripts\python.exe --version)"
} else {
    uv venv --seed --python $PythonVersion .venv; Assert-Success "Creating .venv"
}
uv pip install --python .venv\Scripts\python.exe -r requirements.txt; Assert-Success "Installing requirements.txt"

Write-Step ".env"
if (Test-Path ".env") {
    Write-Host "already exists; left as it is"
} else {
    Copy-Item ".env.example" ".env"
    Write-Host "created from .env.example: fill it in"
}

Write-Host @"

=== Done. Next:
  1. Fill in .env: your YellowDog key, Hugging Face repository and token.
  2. Sign in to AWS, if you haven't:  aws configure sso   (or aws configure)
  3. Activate the environment in each new terminal:  .venv\Scripts\Activate.ps1
     If PowerShell refuses, run this once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  4. Then:  jupyter lab yd-training.ipynb    or    python configure.py --dry-run
"@
