#!/usr/bin/env bash
# ============================================================
# PRISM Application Start Script (macOS & Linux)
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DARKGRAY='\033[1;30m'
NC='\033[0m'

echo ""
echo -e "${CYAN}========================================="
echo -e "          Starting PRISM System          "
echo -e "=========================================${NC}"
echo ""

OS="$(uname -s)"
case "$OS" in
  Darwin*) PLATFORM="macOS" ;;
  Linux*)  PLATFORM="Linux" ;;
  *)       echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PRISM_MODEL="llama3.2:3b"

# ── Ensure venv exists ─────────────────────────────────────────
if [ ! -f "venv/bin/activate" ]; then
  echo -e "${RED}Virtual environment not found! Please run ./install_prism.sh first.${NC}"
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

# ── Check and start Ollama if needed ──────────────────────────
echo -e "${YELLOW}Checking Ollama status...${NC}"
if command -v ollama &>/dev/null; then
  if ! curl -s http://localhost:11434 &>/dev/null; then
    echo -e "${YELLOW}Ollama is not running. Starting Ollama server...${NC}"
    if [ "$PLATFORM" = "macOS" ]; then
      open -a Ollama 2>/dev/null || (ollama serve &>/dev/null &)
    else
      ollama serve &>/dev/null &
    fi
    sleep 4
    echo -e "${GREEN}Ollama server started.${NC}"
  else
    echo -e "${GREEN}Ollama is running.${NC}"
  fi

  # Remove llama3.2-vision if present (incompatible multimodal model)
  if ollama list 2>/dev/null | grep -q "llama3.2-vision"; then
    echo -e "${YELLOW}Removing llama3.2-vision (incompatible multimodal model)...${NC}"
    ollama rm llama3.2-vision 2>/dev/null || true
    echo -e "${GREEN}Removed.${NC}"
  fi

  # Pull model if missing
  if ollama list 2>/dev/null | grep -q -F "$PRISM_MODEL"; then
    echo -e "${GREEN}Model '$PRISM_MODEL' is ready.${NC}"
  else
    echo -e "${YELLOW}Model '$PRISM_MODEL' not found. Pulling now (~2 GB)...${NC}"
    ollama pull "$PRISM_MODEL"
  fi

  # Report GPU status
  if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || true)
    if [ -n "$GPU_NAME" ]; then
      echo -e "${GREEN}GPU detected: $GPU_NAME (Ollama will use GPU automatically).${NC}"
    fi
  elif [ "$PLATFORM" = "macOS" ] && [ "$(uname -m)" = "arm64" ]; then
    echo -e "${GREEN}Apple Silicon detected (Ollama will use Metal acceleration).${NC}"
  else
    echo -e "${YELLOW}No NVIDIA GPU detected (Ollama will run on CPU).${NC}"
  fi
else
  echo -e "${YELLOW}Ollama is not installed. AI report generation will use template fallback.${NC}"
  echo -e "${YELLOW}Run ./install_prism.sh to install Ollama.${NC}"
fi

export OLLAMA_MODEL="$PRISM_MODEL"
export PYTHONPATH="$SCRIPT_DIR"

# ── 1. Start Backend (Orchestrator) ───────────────────────────
echo ""
echo -e "${YELLOW}Starting Backend Services...${NC}"

(PYTHONPATH="$SCRIPT_DIR" OLLAMA_MODEL="$PRISM_MODEL" python orchestrator.py) &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID).${NC}"

# ── Wait for Backend Health ───────────────────────────────────
echo -e "${YELLOW}Waiting for backend services to initialize...${NC}"
BACKEND_READY=false
MAX_RETRIES=25
RETRY=0

while [ "$BACKEND_READY" = false ] && [ "$RETRY" -lt "$MAX_RETRIES" ]; do
  P3_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8005/api/v1/health 2>/dev/null || true)
  P1_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health 2>/dev/null || true)
  if [ "$P3_STATUS" = "200" ] && [ "$P1_STATUS" = "200" ]; then
    BACKEND_READY=true
    break
  fi
  sleep 0.8
  RETRY=$((RETRY + 1))
done

if [ "$BACKEND_READY" = true ]; then
  echo -e "${GREEN}Backend services ready! (Phase 1 :8001, Phase 3 :8005)${NC}"

  # Show hardware profile for TotalSegmentator timeout
  echo ""
  echo -e "${DARKGRAY}  System Profile (used for adaptive Phase 2 timeout):${NC}"
  if command -v nvidia-smi &>/dev/null; then
    GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || true)
    echo -e "${GREEN}    GPU  : $GPU_LINE (CUDA - base timeout 120s)${NC}"
  elif [ "$PLATFORM" = "macOS" ] && [ "$(uname -m)" = "arm64" ]; then
    echo -e "${GREEN}    GPU  : Apple Silicon MPS (base timeout 120s)${NC}"
  else
    echo -e "${YELLOW}    GPU  : None detected (CPU-only - base timeout 600s)${NC}"
  fi

  if [ "$PLATFORM" = "macOS" ]; then
    RAM_TOTAL_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_TOTAL_GB=$((RAM_TOTAL_BYTES / 1024 / 1024 / 1024))
    echo -e "${DARKGRAY}    RAM  : ${RAM_TOTAL_GB} GB total${NC}"
  else
    RAM_FREE_MB=$(free -m 2>/dev/null | awk '/^Mem:/ {print $7}' || echo 0)
    RAM_TOTAL_MB=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 0)
    RAM_FREE_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_FREE_MB/1024}")
    RAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_TOTAL_MB/1024}")
    echo -e "${DARKGRAY}    RAM  : ${RAM_FREE_GB} GB free / ${RAM_TOTAL_GB} GB total${NC}"
  fi
  echo -e "${DARKGRAY}    (Final timeout = base * RAM-scale + volume-size bonus, max 1800s)${NC}"
  echo ""
else
  echo -e "${DARKGRAY}Proceeding with frontend startup...${NC}"
fi

# ── 2. Start Frontend (React/Vite) ────────────────────────────
echo -e "${YELLOW}Starting Frontend (React)...${NC}"
FRONTEND_PID=""

if [ -d "prism-react" ]; then
  (cd prism-react && npm run dev) &
  FRONTEND_PID=$!
  echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID).${NC}"
else
  echo -e "${RED}prism-react directory not found. Cannot start frontend.${NC}"
fi

# Save process IDs to .prism_pids.json for automatic shutdown management
cat <<EOF > .prism_pids.json
{
  "backend_pid": $BACKEND_PID,
  "frontend_pid": ${FRONTEND_PID:-null}
}
EOF
echo -e "${DARKGRAY}Process IDs saved for automatic shutdown management.${NC}"

# Cleanup handler on Ctrl+C or kill
cleanup() {
  echo ""
  echo -e "${YELLOW}Shutting down PRISM services...${NC}"
  if [ -n "$FRONTEND_PID" ]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
  if [ -n "$BACKEND_PID" ]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  rm -f .prism_pids.json
  echo -e "${GREEN}PRISM stopped.${NC}"
  exit 0
}
trap cleanup SIGINT SIGTERM

echo ""
echo -e "${CYAN}========================================="
echo -e "   PRISM is running!"
echo -e "   - Frontend: http://localhost:5173"
echo -e "   - Backend:  http://localhost:8001"
echo -e "   - Phase 3:  http://localhost:8005"
echo -e "   - LLM Model: $PRISM_MODEL"
echo -e "=========================================${NC}"
echo ""

# ── 3. Auto-open browser ──────────────────────────────────────
echo -e "${YELLOW}Opening browser in 2 seconds...${NC}"
sleep 2

if [ "$PLATFORM" = "macOS" ]; then
  open "http://localhost:5173" &>/dev/null || true
else
  if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:5173" &>/dev/null &
  elif command -v google-chrome &>/dev/null; then
    google-chrome "http://localhost:5173" &>/dev/null &
  elif command -v firefox &>/dev/null; then
    firefox "http://localhost:5173" &>/dev/null &
  fi
fi

echo -e "${DARKGRAY}Services will close automatically when the browser tab is closed or when pressing Ctrl+C.${NC}"
echo ""

# Wait on background processes
wait
