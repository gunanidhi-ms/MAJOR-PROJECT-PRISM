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

# Activate venv
$activateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "Failed to find venv activation script." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip --quiet

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
    try {
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        Write-Host "CUDA PyTorch installed successfully." -ForegroundColor Green
    } catch {
        Write-Host "CUDA wheel not available for this Python version. Installing standard PyTorch..." -ForegroundColor Yellow
        pip install torch torchvision
    }
} else {
    Write-Host "No NVIDIA GPU detected. Installing standard CPU PyTorch..." -ForegroundColor Yellow
    pip install torch torchvision
    Write-Host "CPU PyTorch installed." -ForegroundColor Green
}

# 4. Install Python Dependencies
Write-Host "`n[4/6] Installing Python dependencies..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    Write-Host "Installing from root requirements.txt..."
    pip install -r requirements.txt
    Write-Host "Python dependencies installed." -ForegroundColor Green
} else {
    Write-Host "Root requirements.txt not found. Installing per-phase..." -ForegroundColor Yellow
    if (Test-Path "phase1_ingestion\requirements.txt") {
        Write-Host "Installing Phase 1 dependencies..."
        pip install -r phase1_ingestion\requirements.txt
    }
    if (Test-Path "phase2_segmentation\requirements.txt") {
        Write-Host "Installing Phase 2 dependencies..."
        pip install -r phase2_segmentation\requirements.txt
    }
    if (Test-Path "phase3_reporting\requirements.txt") {
        Write-Host "Installing Phase 3 dependencies..."
        pip install -r phase3_reporting\requirements.txt
    }
}

# 5. Check and Setup Ollama
Write-Host "`n[5/6] Setting up Ollama (AI Report Generation)..." -ForegroundColor Yellow

$PRISM_MODEL = "llama3.2:3b"

if (-Not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is NOT installed. Attempting to install via winget..." -ForegroundColor Yellow
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        try {
            winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
            Write-Host "Ollama installed via winget." -ForegroundColor Green
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        } catch {
            Write-Host "winget install failed. Please install Ollama manually from https://ollama.com/" -ForegroundColor Red
            Write-Host "After installing, run: ollama pull $PRISM_MODEL" -ForegroundColor Yellow
        }
    } else {
        Write-Host "winget is not available. Please download Ollama from https://ollama.com/" -ForegroundColor Yellow
        Write-Host "After installing, run: ollama pull $PRISM_MODEL" -ForegroundColor Yellow
    }
}

if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    # Remove incompatible multimodal model if present
    $modelList = (ollama list 2>&1) -join "`n"
    if ($modelList -match "llama3.2-vision") {
        Write-Host "Removing incompatible llama3.2-vision model..." -ForegroundColor Yellow
        ollama rm llama3.2-vision 2>$null
        Write-Host "llama3.2-vision removed." -ForegroundColor Green
    }

    # Check model
    $hasModel = $false
    if ($modelList -match "llama3\.2:3b") {
        $hasModel = $true
    }

    if (-Not $hasModel) {
        Write-Host "Pulling '$PRISM_MODEL' (2.0 GB - recommended for PRISM)..." -ForegroundColor Yellow
        ollama pull $PRISM_MODEL
    } else {
        Write-Host "Model '$PRISM_MODEL' is ready." -ForegroundColor Green
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
