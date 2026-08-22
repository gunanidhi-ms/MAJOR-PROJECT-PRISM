#!/bin/bash
set -e

echo "=========================================="
echo -e "\033[36m   PRISM Autonomous Pipeline Startup      \033[0m"
echo "=========================================="

# 1. Check if python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH."
    exit 1
fi
PYTHON="python3"

# 2. Check for virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo -e "\033[33mActivating virtual environment...\033[0m"
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo -e "\033[33mActivating virtual environment...\033[0m"
    source venv/bin/activate
else
    echo -e "\033[33mWARNING: No virtual environment found (.venv or venv). Proceeding with global Python.\033[0m"
fi

# 3. Check for .env file
if [ ! -f ".env" ]; then
    echo -e "\033[33mNo .env file found at root. Creating from defaults...\033[0m"
    cat << EOF > .env
# Phase 1: Ingestion
DICOM_PORT=11112
WS_PORT=8001

# Phase 2 -> Phase 3 bridge
PHASE3_ENABLED=true
PHASE3_URL=http://localhost:8000/api/v1/generate-report

# Phase 3: Reporting
PHASE3_PORT=8000
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
SKIP_LLM=false
REPORTS_DIR=./storage/reports
EOF
fi

# 4. Start orchestrator
echo -e "\033[32mStarting Orchestrator...\033[0m"
$PYTHON orchestrator.py
