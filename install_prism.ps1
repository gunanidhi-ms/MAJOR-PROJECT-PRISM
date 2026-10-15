# ============================================================
# PRISM Installation Script (Windows PowerShell)
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "       PRISM Installation Script         " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.10+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# 2. Create Virtual Environment
Write-Host "`n[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    Write-Host "Creating new virtual environment 'venv'..."
    python -m venv venv
} else {
    Write-Host "Virtual environment 'venv' already exists." -ForegroundColor Green
}

# Resolve venv paths directly to ensure correct binaries are always called
$venvPython = (Resolve-Path ".\venv\Scripts\python.exe").Path
$venvPip = (Resolve-Path ".\venv\Scripts\pip.exe").Path
$venvScripts = (Resolve-Path ".\venv\Scripts").Path

# Prepend venv\Scripts to session Path
if ($env:Path -notlike "$venvScripts*") {
    $env:Path = "$venvScripts;$env:Path"
}
$env:VIRTUAL_ENV = (Resolve-Path ".\venv").Path

# Activate venv if script exists
$activateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    try { & $activateScript } catch { }
}

# Upgrade pip using explicit venv python
Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip --quiet

# 3. Detect GPU and Install PyTorch (CUDA if GPU present, else CPU)
Write-Host "`n[3/6] Detecting GPU for PyTorch / TotalSegmentator acceleration..." -ForegroundColor Yellow

$gpuDetected = $false
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    try {
        $smiOutput = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
        if ($smiOutput) {
            Write-Host "NVIDIA GPU detected: $smiOutput" -ForegroundColor Green
            $gpuDetected = $true
        }
    } catch {
        Write-Host "Could not query GPU details." -ForegroundColor Yellow
    }
}

if ($gpuDetected) {
    Write-Host "Attempting CUDA PyTorch installation (cu124)..." -ForegroundColor Green
    & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA wheel not available for this Python version. Installing standard PyTorch..." -ForegroundColor Yellow
        & $venvPip install torch torchvision
    } else {
        Write-Host "CUDA PyTorch installed successfully." -ForegroundColor Green
    }
} else {
    Write-Host "No NVIDIA GPU detected. Installing standard CPU PyTorch..." -ForegroundColor Yellow
    & $venvPip install torch torchvision
    Write-Host "CPU PyTorch installed." -ForegroundColor Green
}

# 4. Install Python Dependencies
Write-Host "`n[4/6] Installing Python dependencies..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    Write-Host "Installing from root requirements.txt..."
    & $venvPip install -r requirements.txt
    Write-Host "Python dependencies installed." -ForegroundColor Green
} else {
    Write-Host "Root requirements.txt not found. Installing per-phase..." -ForegroundColor Yellow
    if (Test-Path "phase1_ingestion\requirements.txt") {
        Write-Host "Installing Phase 1 dependencies..."
        & $venvPip install -r phase1_ingestion\requirements.txt
    }
    if (Test-Path "phase2_segmentation\requirements.txt") {
        Write-Host "Installing Phase 2 dependencies..."
        & $venvPip install -r phase2_segmentation\requirements.txt
    }
    if (Test-Path "phase3_reporting\requirements.txt") {
        Write-Host "Installing Phase 3 dependencies..."
        & $venvPip install -r phase3_reporting\requirements.txt
    }
}

# 5. Check and Setup Ollama
Write-Host "`n[5/6] Setting up Ollama (AI Report Generation)..." -ForegroundColor Yellow

$PRISM_MODEL = "llama3.2:3b"
$ollamaDefaultDir = "$env:LOCALAPPDATA\Programs\Ollama"

# 5.1 Locate or add Ollama to current session PATH
if (Test-Path "$ollamaDefaultDir\ollama.exe") {
    if ($env:Path -notlike "*$ollamaDefaultDir*") {
        $env:Path = "$ollamaDefaultDir;$env:Path"
    }
}

# 5.2 Install Ollama if not found
if (-Not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is NOT installed. Attempting to install via winget..." -ForegroundColor Yellow
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        try {
            # Run winget with Start-Process so it doesn't deadlock the PowerShell console pipes
            $wingetArgs = "install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements --silent"
            $p = Start-Process -FilePath "winget" -ArgumentList $wingetArgs -Wait -PassThru -NoNewWindow
            
            # The Ollama Windows installer spawns: cmd.exe /c "ollama app.exe" --hide --fast-startup
            # Because cmd.exe runs synchronously without 'start', it holds the terminal open forever.
            # Terminate the waiting cmd.exe wrapper (leaves the background ollama app running).
            Start-Sleep -Seconds 2
            Get-CimInstance Win32_Process | Where-Object { 
                $_.Name -eq "cmd.exe" -and $_.CommandLine -like "*ollama app.exe*" 
            } | ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }

            if (Test-Path "$ollamaDefaultDir\ollama.exe") {
                if ($env:Path -notlike "*$ollamaDefaultDir*") {
                    $env:Path = "$ollamaDefaultDir;$env:Path"
                }
                Write-Host "Ollama installed via winget." -ForegroundColor Green
            }
        } catch {
            Write-Host "winget install failed." -ForegroundColor Red
        }
    }

    if (-Not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
        Write-Host "Please download and install Ollama manually from: https://ollama.com/" -ForegroundColor Red
        Write-Host "After installing, re-run: .\install_prism.ps1" -ForegroundColor Yellow
    }
}

# 5.3 Verify Ollama server daemon is active and responding
if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    Write-Host "Checking Ollama server connection..." -ForegroundColor Yellow
    $serverRunning = $false
    try {
        $health = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($health.StatusCode -eq 200) { $serverRunning = $true }
    } catch {}

    if (-Not $serverRunning) {
        Write-Host "Ollama server is not running. Starting background service..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        $retries = 0
        while (-Not $serverRunning -and $retries -lt 10) {
            Start-Sleep -Seconds 1
            $retries++
            try {
                $health = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
                if ($health.StatusCode -eq 200) { $serverRunning = $true }
            } catch {}
        }
    }

    if ($serverRunning) {
        Write-Host "Ollama server is active." -ForegroundColor Green
    } else {
        Write-Host "Warning: Ollama server did not respond on http://localhost:11434." -ForegroundColor Yellow
    }

    # 5.4 Remove incompatible multimodal model if present
    $modelList = ""
    try {
        $modelList = (ollama list 2>&1) -join "`n"
    } catch {}

    if ($modelList -match "llama3.2-vision") {
        Write-Host "Removing incompatible llama3.2-vision model..." -ForegroundColor Yellow
        & ollama rm llama3.2-vision 2>$null
        Write-Host "llama3.2-vision removed." -ForegroundColor Green
    }

    # 5.5 Check or pull recommended PRISM model
    if ($modelList -match "llama3\.2:3b") {
        Write-Host "Model '$PRISM_MODEL' is ready." -ForegroundColor Green
    } else {
        Write-Host "Pulling '$PRISM_MODEL' (2.0 GB - recommended for PRISM)..." -ForegroundColor Yellow
        Write-Host "Downloading model weights from Ollama library (this may take a few minutes)..." -ForegroundColor DarkGray
        & ollama pull $PRISM_MODEL
        Write-Host "Model '$PRISM_MODEL' downloaded successfully." -ForegroundColor Green
    }
}

# 6. Install Frontend Dependencies
Write-Host "`n[6/6] Installing frontend dependencies..." -ForegroundColor Yellow
if (Get-Command "npm" -ErrorAction SilentlyContinue) {
    Push-Location prism-react
    npm install
    Pop-Location
    Write-Host "Frontend dependencies installed." -ForegroundColor Green
} else {
    Write-Host "npm is not installed. Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    Write-Host "Frontend will not be able to run without Node.js." -ForegroundColor Yellow
}

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "       PRISM Installation Complete       " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
if ($gpuDetected) {
    Write-Host "GPU acceleration: ENABLED (Ollama GPU offload + NVIDIA GPU detected)" -ForegroundColor Green
} else {
    Write-Host "GPU acceleration: CPU mode (no NVIDIA GPU found)" -ForegroundColor Yellow
}
Write-Host "LLM model:        $PRISM_MODEL" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run PRISM by executing: .\start_prism.ps1" -ForegroundColor Green
