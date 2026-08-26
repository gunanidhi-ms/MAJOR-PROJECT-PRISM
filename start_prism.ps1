<#
.SYNOPSIS
Starts the PRISM application (Frontend and Backend components).

.DESCRIPTION
This script starts:
1. Orchestrator Backend (FastAPI, DICOM Listener, Subprocesses) on port 8001/8005
2. React Frontend via Vite on port 5173
3. Opens browser to http://localhost:5173 after services initialize

GPU Acceleration:
- Ollama auto-detects your NVIDIA GPU and offloads model layers to VRAM.
- TotalSegmentator uses GPU if CUDA PyTorch was installed via install_prism.ps1.
#>

$ErrorActionPreference = "Stop"
$InformationPreference = "Continue"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "          Starting PRISM System          " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Recommended model: llama3.2:3b
# - 2.0 GB — fits entirely in 4 GB VRAM (RTX 3050 Ti, 3060, etc.)
# - Ollama auto-offloads to GPU if NVIDIA GPU is present
# - Falls back to CPU automatically if no GPU is found
# - 2-3s report generation on GPU, 6-10s on CPU
$PRISM_MODEL = "llama3.2:3b"

# Ensure venv exists
if (-Not (Test-Path "venv")) {
    Write-Host "Virtual environment not found! Please run .\install_prism.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate venv for Ollama health check
& .\venv\Scripts\Activate.ps1

# Check and start Ollama if needed
Write-Host "Checking Ollama status..." -ForegroundColor Yellow
if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    try {
        $ollamaHealth = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 3 -UseBasicParsing -ErrorAction SilentlyContinue
        Write-Host "Ollama is running." -ForegroundColor Green
    } catch {
        Write-Host "Ollama is not running. Starting Ollama server..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        Write-Host "Ollama server started." -ForegroundColor Green
    }

    # Ensure correct model is available; remove incompatible ones
    $modelList = (ollama list 2>&1) -join "`n"

    # Remove llama3.2-vision if present (multimodal mllama arch — crashes text-only chat)
    if ($modelList -match "llama3.2-vision") {
        Write-Host "Removing llama3.2-vision (incompatible multimodal model)..." -ForegroundColor Yellow
        ollama rm llama3.2-vision 2>$null
        Write-Host "Removed." -ForegroundColor Green
    }

    # Check if llama3.2:3b is already present in ollama list
    if ($modelList -match "llama3\.2:3b") {
        Write-Host "Model '$PRISM_MODEL' is ready." -ForegroundColor Green
    } else {
        Write-Host "Model '$PRISM_MODEL' not found. Pulling now (~2 GB)..." -ForegroundColor Yellow
        ollama pull $PRISM_MODEL
    }

    # Report GPU status
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
        Write-Host "GPU detected: $gpuName (Ollama will use GPU automatically)." -ForegroundColor Green
    } else {
        Write-Host "No NVIDIA GPU detected (Ollama will run on CPU)." -ForegroundColor Yellow
    }
} else {
    Write-Host "Ollama is not installed. AI report generation will use template fallback." -ForegroundColor Yellow
    Write-Host "Run .\install_prism.ps1 to install Ollama." -ForegroundColor Yellow
}

# Set OLLAMA_MODEL so Phase 3 picks it up from the environment
$env:OLLAMA_MODEL = $PRISM_MODEL

# 1. Start Backend (Orchestrator)
Write-Host "`nStarting Backend Services..." -ForegroundColor Yellow

$backendProcess = Start-Process -FilePath "powershell" `
    -ArgumentList "-NoExit", "-Command", "& { .\venv\Scripts\Activate.ps1; `$env:PYTHONPATH='.'; `$env:OLLAMA_MODEL='$PRISM_MODEL'; python orchestrator.py }" `
    -PassThru -WindowStyle Normal
Write-Host "Backend started in new window (PID: $($backendProcess.Id))." -ForegroundColor Green

# Wait until Phase 1 & Phase 3 backends are fully listening and healthy
Write-Host "Waiting for backend services to initialize..." -ForegroundColor Yellow
$backendReady = $false
$maxRetries = 25
$retry = 0

while (-not $backendReady -and $retry -lt $maxRetries) {
    try {
        $p3 = Invoke-WebRequest -Uri "http://127.0.0.1:8005/api/v1/health" -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        $p1 = Invoke-WebRequest -Uri "http://127.0.0.1:8001/health" -TimeoutSec 1 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($p3.StatusCode -eq 200 -and $p1.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    } catch {}
    Start-Sleep -Milliseconds 800
    $retry++
}

if ($backendReady) {
    Write-Host "Backend services ready! (Phase 1 :8001, Phase 3 :8005)" -ForegroundColor Green

    # Show hardware profile for TotalSegmentator timeout
    Write-Host ""
    Write-Host "  System Profile (used for adaptive Phase 2 timeout):" -ForegroundColor DarkGray
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        $gpuLine = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
        Write-Host "    GPU  : $gpuLine (CUDA - base timeout 120s)" -ForegroundColor Green
    } else {
        Write-Host "    GPU  : None detected (CPU-only - base timeout 600s)" -ForegroundColor Yellow
    }
    try {
        $ramFreeGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
        $ramTotalGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB, 1)
        Write-Host "    RAM  : $ramFreeGB GB free / $ramTotalGB GB total" -ForegroundColor DarkGray
    } catch {
        Write-Host "    RAM  : (could not query)" -ForegroundColor DarkGray
    }
    Write-Host "    (Final timeout = base * RAM-scale + volume-size bonus, max 1800s)" -ForegroundColor DarkGray
    Write-Host ""
} else {
    Write-Host "Proceeding with frontend startup..." -ForegroundColor DarkGray
}


# 2. Start Frontend
Write-Host "Starting Frontend (React)..." -ForegroundColor Yellow

if (Test-Path "prism-react") {
    $frontendProcess = Start-Process -FilePath "powershell" `
        -ArgumentList "-NoExit", "-Command", "& { cd prism-react; npm run dev }" `
        -PassThru -WindowStyle Normal
    Write-Host "Frontend started in new window (PID: $($frontendProcess.Id))." -ForegroundColor Green
} else {
    Write-Host "prism-react directory not found. Cannot start frontend." -ForegroundColor Red
}

# Save process IDs to .prism_pids.json for clean auto-close on browser tab close
$pidsObj = @{
    backend_pid = $backendProcess.Id
    frontend_pid = if ($frontendProcess) { $frontendProcess.Id } else { $null }
}
$pidsObj | ConvertTo-Json | Set-Content -Path ".prism_pids.json" -Encoding UTF8
Write-Host "Process IDs saved for automatic shutdown management." -ForegroundColor DarkGray

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "   PRISM is running!" -ForegroundColor Cyan
Write-Host "   - Frontend: http://localhost:5173"
Write-Host "   - Backend:  http://localhost:8001"
Write-Host "   - Phase 3:  http://localhost:8005"
Write-Host "   - LLM Model: $PRISM_MODEL"
Write-Host "=========================================" -ForegroundColor Cyan

# 3. Auto-open browser
Write-Host "`nOpening browser in 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
Start-Process "http://localhost:5173"

Write-Host "`nServices will close automatically when the browser tab is closed or when clicking Shutdown." -ForegroundColor DarkGray
