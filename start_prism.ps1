<#
.SYNOPSIS
Starts the full PRISM Autonomous End-to-End Pipeline.
Includes automatic Ollama startup and readiness check.
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   PRISM Autonomous Pipeline Startup      " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check if python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# 2. Check for virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Yellow
    . ".venv\Scripts\Activate.ps1"
    Write-Host "  [OK] Virtual environment (.venv) activated." -ForegroundColor Green
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (venv)..." -ForegroundColor Yellow
    . "venv\Scripts\Activate.ps1"
    Write-Host "  [OK] Virtual environment (venv) activated." -ForegroundColor Green
} else {
    Write-Host "WARNING: No virtual environment found (.venv or venv). Proceeding with global Python." -ForegroundColor DarkYellow
}

# 3. Check for .env file
if (-not (Test-Path ".env")) {
    Write-Host "No .env file found at root. Creating from defaults..." -ForegroundColor Yellow
    @"
# Phase 1: Ingestion
DICOM_PORT=11112
WS_PORT=8001

# Phase 2 -> Phase 3 bridge
PHASE3_ENABLED=true
PHASE3_URL=http://localhost:8000/api/v1/generate-report

# Phase 3: Reporting
PHASE3_PORT=8000
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral:7b
SKIP_LLM=false
REPORTS_DIR=./storage/reports
"@ | Out-File -FilePath ".env" -Encoding utf8
}

# 4. Start / ensure Ollama is running
Write-Host "" -ForegroundColor Green
Write-Host "--- Step 1: Checking Ollama ---" -ForegroundColor Cyan
$ollamaReady = $false
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($resp.StatusCode -eq 200) {
        Write-Host "  [OK] Ollama already running on port 11434." -ForegroundColor Green
        $ollamaReady = $true
    }
} catch {
    Write-Host "  Ollama not running - starting it in background..." -ForegroundColor Yellow
}

if (-not $ollamaReady) {
    if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
        Start-Process powershell -ArgumentList "-NoExit -Command `"ollama serve`"" -WindowStyle Minimized
        Write-Host "  Waiting for Ollama to become ready (up to 30s)..." -ForegroundColor Yellow
        $waited = 0
        while ($waited -lt 30) {
            Start-Sleep -Seconds 2
            $waited += 2
            try {
                $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 1 -ErrorAction SilentlyContinue
                if ($resp.StatusCode -eq 200) {
                    Write-Host "  [OK] Ollama is ready after ${waited}s." -ForegroundColor Green
                    $ollamaReady = $true
                    break
                }
            } catch { }
        }
        if (-not $ollamaReady) {
            Write-Host "  [WARN] Ollama did not respond in time - reports will fall back to template." -ForegroundColor DarkYellow
        }
    } else {
        Write-Host "  [WARN] 'ollama' command not found - install from https://ollama.com. Falling back to template." -ForegroundColor DarkYellow
    }
}

# 5. Start Frontend Dashboard
Write-Host "" -ForegroundColor Green
Write-Host "--- Step 2: Starting Frontend ---" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -NoExit -Command `"cd prism-react; npm.cmd run dev`"" -WindowStyle Normal
Write-Host "  [OK] Frontend starting at http://localhost:5173" -ForegroundColor Green

# 6. Start DICOM Replay Sender (delayed, so pipeline is ready first)
Write-Host "" -ForegroundColor Green
Write-Host "--- Step 3: Scheduling DICOM Sender (after 15s delay) ---" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -NoExit -Command `"Start-Sleep -Seconds 15; if (Test-Path .venv\Scripts\activate.ps1) { . .venv\Scripts\activate.ps1 }; python -m phase1_ingestion.replay_sender`"" -WindowStyle Normal
Write-Host "  [OK] DICOM Replay Sender scheduled to start in 15 seconds." -ForegroundColor Green

# 7. Start Orchestrator (blocking - keeps this window alive)
Write-Host "" -ForegroundColor Green
Write-Host "--- Step 4: Starting Orchestrator (Phase 1 + 2 + 3) ---" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop the entire pipeline." -ForegroundColor DarkGray
Write-Host "" -ForegroundColor Green
python orchestrator.py
